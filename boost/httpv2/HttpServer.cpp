#include "HttpServer.h"

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/strand.hpp>
#include <boost/config.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace bhttp {
using namespace boost;

static void InternalLog(int log_level, const std::string& content)
{
    (void)log_level;
    (void)content;
}

LogFunc g_Log = &InternalLog;

void SetLogFunc(LogFunc func)
{
    if (func)
        g_Log = std::move(func);
}

void LogDebug(const std::string& content)
{
    if (g_Log)
        g_Log(ELogLevel::EDebug, content);
}

void LogWarning(const std::string& content)
{
    if (g_Log)
        g_Log(ELogLevel::EWarning, content);
}

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
        void operator()(beast::http::message<isRequest, Body, Fields>&& msg) const
        {
            // The lifetime of the message has to extend
            // for the duration of the async operation so
            // we use a shared_ptr to manage it.
            auto sp = std::make_shared<
                beast::http::message<isRequest, Body, Fields>>(std::move(msg));

            // Store a type-erased version of the shared
            // pointer in the class to keep it alive.
            self_.res_ = sp;

            // Write the response
            beast::http::async_write(
                self_.stream_,
                *sp,
                beast::bind_front_handler(
                    &Session::on_write,
                    self_.shared_from_this(),
                    sp->need_eof()));
        }
    };

    beast::tcp_stream stream_;
    beast::flat_buffer buffer_;
    std::shared_ptr<std::string const> doc_root_;
    beast::http::request<beast::http::string_body> req_;
    std::shared_ptr<void> res_;
    send_lambda lambda_;
    HttpServer& http_server_;

public:
    // Take ownership of the stream
    Session(asio::ip::tcp::socket&& socket, std::shared_ptr<std::string const> const& doc_root, HttpServer& server)
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
        beast::http::async_read(stream_, buffer_, req_,
            beast::bind_front_handler(
                &Session::on_read,
                shared_from_this()));
    }

    void on_read(beast::error_code ec, std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // This means they closed the connection
        if(ec == beast::http::error::end_of_stream)
            return do_close();

        if (ec)
            return;
        // Send the response
        //handle_request(*doc_root_, std::move(req_), lambda_);
        http_server_.HandleRequest(std::move(req_), lambda_);
    }

    void on_write(bool close, beast::error_code ec, std::size_t bytes_transferred)
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
        beast::error_code ec;
        stream_.socket().shutdown(asio::ip::tcp::socket::shutdown_send, ec);

        // At this point the connection is closed gracefully
    }
};

/**
 * 
 ************************************************************************/

class Listener : public std::enable_shared_from_this<Listener>
{
    boost::asio::io_context&            ioc_;
    boost::asio::ip::tcp::acceptor      acceptor_;
    std::shared_ptr<std::string const>  doc_root_;
    boost::asio::ip::tcp::endpoint      endpoint_;
    HttpServer&                         http_server_;

public:
    Listener(boost::asio::io_context& ioc, boost::asio::ip::tcp::endpoint endpoint, std::shared_ptr<std::string const> const& doc_root, HttpServer& server)
        : ioc_(ioc),
        acceptor_(boost::asio::make_strand(ioc)),
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
            LogWarning("http acceptor bind endpoint failed. " + std::to_string(ec.value()) + " "  + ec.message());
            return false;
        }

        acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
        if (ec) {
            LogWarning("http acceptor listen failed.");
            return false;
        }
        Run();
        return true;
    }

    void Run()
    {
        DoAccept();
    }

    void Shutdown()
    {
        boost::system::error_code ec;
        acceptor_.close(ec);
    }

private:
    void DoAccept()
    {
        acceptor_.async_accept(boost::asio::make_strand(ioc_),
            boost::beast::bind_front_handler(&Listener::OnAccept,shared_from_this()));
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
    m_ioc(),
    m_work_guard(), 
    m_listener(),
    m_thread_pool()
{
}

HttpServer::~HttpServer()
{
    if (m_work_guard)
        m_work_guard->reset();
    if (m_ioc) {
        m_ioc->stop();
    }

    for (auto& t : m_thread_pool) {
        if (t.joinable())
            t.join();
    }
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

bool HttpServer::Init(int N)
{
    m_ioc = std::make_shared<boost::asio::io_context>(N);
    m_work_guard = std::make_shared<WorkGuard>(boost::asio::make_work_guard(*m_ioc));

    auto address = boost::asio::ip::make_address(m_host);
    boost::asio::ip::tcp::endpoint ep{ address, m_port };
    m_listener = std::make_shared<Listener>(*m_ioc, ep, std::make_shared<std::string>("./"), *this);

    if (!m_listener->Init()) {
        LogWarning("listener init failed!");
        return false;
    }

    m_thread_pool.resize(N);
    for (int i = 0; i != N; ++i) {
        m_thread_pool.emplace_back([this]
        {
            while (1) {
                try {
                    m_ioc->run();
                    break;
                } catch (const std::exception& e) {
                    LogWarning("thread catch exception: " + std::string(e.what()));
                    m_ioc->restart();
                }
            }
        });
    }
    return true;
}

void HttpServer::Shutdown()
{
    m_listener->Shutdown();
    m_work_guard.reset();
    m_ioc->stop();
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
    if (m_work_guard) {
        m_work_guard->reset();
        m_work_guard = nullptr;
    }
    if (m_ioc) {
        m_ioc->stop();
        m_ioc = nullptr;
    }
    m_thread_pool.clear();
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
