#include "HttpServer.h"

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>

namespace bhttp {

static LogCallback g_log = nullptr;

const std::string& BHttpVersion()
{
    static const std::string v = "bhttp/0.1";
    return v;
}

static int Vsnprintf(char* buf, std::size_t buflen, const char* format, va_list ap)
#ifdef __GNUC__
__attribute__((format(printf, 3, 0)))
#endif
{
    int r;
    if (!buflen)
        return 0;
#if defined(_MSC_VER) || defined(_WIN32)
    r = _vsnprintf_s(buf, buflen, buflen, format, ap);
    if (r < 0)
        r = _vscprintf(format, ap);
    r = vsnprintf(buf, buflen, format, ap);
#endif
    buf[buflen - 1] = '\0';
    return r;
}

static void FlushLog(int severity,  const char* msg)
{
    if (g_log)
        g_log(severity, msg);
}

static void PrintfLogv(int severity, const char* fmt, va_list ap)
{
    char buf[1024];
    if (fmt)
        Vsnprintf(buf, sizeof(buf), fmt, ap);
    else
        buf[0] = '\0';
    FlushLog(severity, buf);
}

void SetLogCallback(LogCallback func)
{
    g_log = func;
}

void LogDebug(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    PrintfLogv(ESeverity::EDebug, fmt, ap);
    va_end(ap);
}

void LogWarning(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    PrintfLogv(ESeverity::EWarning, fmt, ap);
    va_end(ap);
}

class IOContext
{
public:
    IOContext(std::int32_t index)
        : m_index(index)
        , m_thread()
        , m_ioctx()
        , m_work(boost::asio::make_work_guard(m_ioctx))
    {
        std::thread temp(std::bind(&IOContext::Run, this));
        m_thread = std::move(temp);
    }

    ~IOContext()
    {
        m_work.reset();
        if (!m_ioctx.stopped())
            m_ioctx.stop();
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }

    std::int32_t                        m_index;
    std::thread                         m_thread;
    boost::asio::io_context             m_ioctx;
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_work;

    void Stop()
    {
        m_work.reset();
        if (!m_ioctx.stopped())
            m_ioctx.stop();
    }

private:
    void Run()
    {
        while (1) {
            try {
                m_ioctx.run();
                LogDebug("io_context: %d stop running", m_index);
                break;
            } catch (const std::exception& e) {
                LogWarning("io_context: %d exception: %s ... io_context restart", m_index, e.what());
                m_ioctx.restart();
            }
        }
    }
};
using IOContextPtr = std::shared_ptr<IOContext>;


class IOContextPool
{
public:
    IOContextPool()
        : m_next_index()
        , m_ioc_vec()
    {
    }

    ~IOContextPool()
    {
    }

    void Init(std::int32_t count)
    {
        if (count <= 0)
            count = 1;
        for (std::int32_t i = 0; i != count; ++i) {
            auto ioctx = std::make_shared<IOContext>(i);
            m_ioc_vec.emplace_back(ioctx);
        }
    }

    void Stop()
    {
        for (auto& ioc : m_ioc_vec) {
            ioc->Stop();
        }
    }

    IOContextPtr NextIOContext()
    {
        if (m_ioc_vec.empty())
            return nullptr;
        auto idx = (++m_next_index) % m_ioc_vec.size();
        return m_ioc_vec[idx];
    }

private:
    std::atomic<std::uint64_t>      m_next_index;
    std::vector<IOContextPtr>       m_ioc_vec;
};


class Session : public std::enable_shared_from_this<Session>
{
    // This is the C++11 equivalent of a generic lambda.
    // The function object is used to send an HTTP message.
    struct send_lambda
    {
        Session& self_;

        explicit send_lambda(Session& self)
            : self_(self)
        {
        }

        template<bool isRequest, class Body, class Fields>
        void operator()(boost::beast::http::message<isRequest, Body, Fields>&& msg) const
        {
            // The lifetime of the message has to extend
            // for the duration of the async operation so
            // we use a shared_ptr to manage it.
            auto sp = std::make_shared<boost::beast::http::message<isRequest, Body, Fields>>(std::move(msg));

            // Store a type-erased version of the shared
            // pointer in the class to keep it alive.
            self_.res_ = sp;

            // Write the response
            boost::beast::http::async_write(
                self_.stream_,
                *sp,
                boost::beast::bind_front_handler(
                    &Session::on_write,
                    self_.shared_from_this(),
                    sp->need_eof()));
        }
    };

    boost::beast::tcp_stream stream_;
    boost::beast::flat_buffer buffer_;
    std::shared_ptr<std::string const> doc_root_;
    boost::beast::http::request<boost::beast::http::string_body> req_;
    std::shared_ptr<void> res_;
    send_lambda lambda_;
    HttpServer& http_server_;

public:
    // Take ownership of the stream
    Session(boost::asio::ip::tcp::socket&& socket, std::shared_ptr<std::string const> const& doc_root, HttpServer& server)
        : stream_(std::move(socket))
        , doc_root_(doc_root)
        , lambda_(*this),
        http_server_(server)
    {
    }

    void run()
    {
        do_read();
    }

    void do_read()
    {
        // Make the request empty before reading,
        // otherwise the operation behavior is undefined.
        req_ = {};

        // Set the timeout.
        stream_.expires_after(std::chrono::seconds(30));

        // Read a request
        boost::beast::http::async_read(stream_, buffer_, req_,
            boost::beast::bind_front_handler(
                &Session::on_read,
                shared_from_this()));
    }

    void on_read(boost::beast::error_code ec, std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // This means they closed the connection
        if(ec == boost::beast::http::error::end_of_stream)
            return do_close();

        if (ec)
            return;
        // Send the response
        //handle_request(*doc_root_, std::move(req_), lambda_);
        http_server_.HandleRequest(std::move(req_), lambda_);
    }

    void on_write(bool close, boost::beast::error_code ec, std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        if (ec)
            return;

        if(close) {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            return do_close();
        }

        // We're done with the response so delete it
        res_ = nullptr;

        // Read another request
        do_read();
    }

    void do_close()
    {
        // Send a TCP shutdown
        boost::beast::error_code ec;
        stream_.socket().shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);

        // At this point the connection is closed gracefully
    }
};

/**
 * 
 ************************************************************************/

class Listener : public std::enable_shared_from_this<Listener>
{
    boost::asio::ip::tcp::acceptor      acceptor_;
    std::shared_ptr<std::string const>  doc_root_;
    boost::asio::ip::tcp::endpoint      endpoint_;
    HttpServer&                         http_server_;

public:
    Listener(boost::asio::io_context& ioc, boost::asio::ip::tcp::endpoint endpoint, std::shared_ptr<std::string const> const& doc_root, HttpServer& server)
        : acceptor_(ioc),
        doc_root_(doc_root),
        endpoint_(endpoint),
        http_server_(server)
    {
    }

    ~Listener()
    {
    }

    bool Init()
    {
        boost::system::error_code ec;
        acceptor_.open(endpoint_.protocol(), ec);
        if (ec) {
            LogWarning("http acceptor open endpoint failed.");
            return false;
        }
        acceptor_.set_option(boost::asio::socket_base::reuse_address(true), ec);
        if (ec) {
            LogWarning("http acceptor set socket reuse address failed.");
            return false;
        }
        acceptor_.bind(endpoint_, ec);
        if (ec) {
            LogWarning("http acceptor bind endpoint failed. ecode: %d  emsg: %s",  ec.value(), ec.message());
            return false;
        }

        acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
        if (ec) {
            LogWarning("http acceptor listen failed.");
            return false;
        }

        DoAccept();
        return true;
    }

    void Shutdown()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
    }

private:
    void DoAccept()
    {
        auto sid = detail::ThreadID();
        LogDebug("DoAccept thread %s", sid.c_str());

        auto ioc = http_server_.GetIOContextPool().NextIOContext();
        //acceptor_.async_accept(boost::asio::make_strand(ioc->m_ioctx), boost::beast::bind_front_handler(&Listener::OnAccept,shared_from_this()));
        acceptor_.async_accept(ioc->m_ioctx, boost::beast::bind_front_handler(&Listener::OnAccept,shared_from_this()));
    }

    void OnAccept(boost::system::error_code ec, boost::asio::ip::tcp::socket socket)
    {
        if (ec) {
            LogWarning("OnAccept failed");
        } else {
            std::make_shared<Session>(std::move(socket), doc_root_, http_server_)->run();
        }
        DoAccept();
    }
};


/**
 * 
 ************************************************************************/

HttpServer::HttpServer()
    : m_resourcesMap(),
    m_host(),
    m_port(),
    m_io_pool(std::make_shared<IOContextPool>()),
    m_listener_pool(std::make_shared<IOContextPool>()),
    m_listener()
{
}

HttpServer::~HttpServer()
{
}

void HttpServer::SetHost(std::string host)
{
    m_host = std::move(host);
}

void HttpServer::SetPort(std::uint16_t port)
{
    m_port = port;
}

void HttpServer::AddHttpHandler(const std::string& method, const std::string& path, HttpHandler hdl)
{
    m_resourcesMap[method][path] = std::move(hdl);
}

bool HttpServer::Init(int worker_thread)
{
    m_io_pool->Init(worker_thread);
    m_listener_pool->Init(1);
    auto address = boost::asio::ip::make_address(m_host);
    boost::asio::ip::tcp::endpoint ep{ address, m_port };

    auto ioc = m_listener_pool->NextIOContext();
    m_listener = std::make_shared<Listener>(ioc->m_ioctx, ep, std::make_shared<std::string>("./"), *this);

    if (!m_listener->Init()) {
        LogWarning("listener init failed!");
        return false;
    }
    return true;
}

void HttpServer::Shutdown()
{
    m_listener->Shutdown();
    m_listener_pool->Stop();
    m_io_pool->Stop();
}

void HttpServer::Reset()
{
    m_resourcesMap.clear();
    //m_host.clear();
    //m_port = 0;
    if (m_listener) {
        m_listener->Shutdown();
        m_listener = nullptr;
    }
}

IOContextPool& HttpServer::GetIOContextPool()
{
    return *m_io_pool;
}

} // namespace bhttp


/**
 * 
 ************************************************************************/

/*
int main(int argc, char* argv[])
{
    using namespace boost;

    // Check command line arguments.
    if (argc != 5)
    {
        std::cerr <<
            "Usage: http-server-async <address> <port> <doc_root> <threads>\n" <<
            "Example:\n" <<
            "    http-server-async 0.0.0.0 8080 . 1\n";
        return EXIT_FAILURE;
    }
    auto const address = asio::ip::make_address(argv[1]);
    auto const port = static_cast<unsigned short>(std::atoi(argv[2]));
    auto const doc_root = std::make_shared<std::string>(argv[3]);
    auto const threads = std::max<int>(1, std::atoi(argv[4]));

    // The io_context is required for all I/O
    asio::io_context ioc{threads};

    // Create and launch a listening port
    std::make_shared<bhttp::Listener>(
        ioc,
        asio::ip::tcp::endpoint{address, port},
        doc_root)->Init();

    // Run the I/O service on the requested number of threads
    std::vector<std::thread> v;
    v.reserve(threads - 1);
    for(auto i = threads - 1; i > 0; --i)
        v.emplace_back(
        [&ioc]
        {
            ioc.run();
        });
    ioc.run();

    return EXIT_SUCCESS;
}
*/


namespace bhttp {
namespace detail {

std::string ThreadID()
{
    std::ostringstream ostm{};
    ostm << std::this_thread::get_id();
    return ostm.str();
}

} // namespace detail
} // namespace detail 

