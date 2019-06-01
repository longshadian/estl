#include "HttpServer.h"

#include "server_certificate.hpp"
#include "ssl_stream.hpp"

#include "HttpSession.h"

HttpServer::HttpServer()
    : m_doc_root("/")
    , m_host("0.0.0.0")
    , m_port(10086)
    , m_read_timeout_milliseconds(std::chrono::milliseconds{15 * 1000})
    , m_resourcesMap()
    , m_thread_pool()
    , m_io_context()
    , m_ssl_context()
    , m_listner()
{
}

HttpServer::~HttpServer()
{
    if (!m_io_context->stopped()) {
        Stop();
        WaitExit();
    }
}

void HttpServer::InitHost(std::string host)
{
    m_host = std::move(host);
}

void HttpServer::InitPort(uint16_t port)
{
    m_port = port;
}

void HttpServer::InitDocRoot(std::string doc_root)
{
    m_doc_root = std::move(doc_root);
}

void HttpServer::InitResource(std::vector<Resource> resource_array)
{
    for (auto& it : resource_array) {
        m_resourcesMap[it.m_path][it.m_method] = std::move(it.m_handler);
    }
}

void HttpServer::InitReadTimeoutMilliseconds(std::chrono::milliseconds millisec)
{
    m_read_timeout_milliseconds = millisec;
}

void HttpServer::InitSslContext()
{
    // The SSL context is required, and holds certificates
    m_ssl_context = std::make_unique<boost::asio::ssl::context>(boost::asio::ssl::context::sslv23);

    // This holds the self-signed certificate used by the server
    load_server_certificate(*m_ssl_context);
}

bool HttpServer::Start(int thread_num)
{
    const auto threads = std::max<int>(1, thread_num);
    m_io_context = std::make_unique<boost::asio::io_context>(threads);

    ssl::context ctx{ ssl::context::sslv23 };

    boost::asio::ip::address address{};
    address.from_string(m_host);

    m_listner = std::make_shared<listener>(*this, boost::asio::ip::tcp::endpoint{address, m_port});
    m_listner->run();

    m_thread_pool.reserve(threads);
    for (int i = 0; i != threads; ++i) {
        m_thread_pool.emplace_back([this] { m_io_context->run(); });
    }
    return true;
}

void HttpServer::Stop()
{
    m_io_context->stop();
}

void HttpServer::WaitExit()
{
    for (auto& t : m_thread_pool) {
        if (t.joinable())
            t.join();
    }
}

const std::string& HttpServer::GetDocRoot() const
{
    return m_doc_root;
}

boost::asio::io_context& HttpServer::GetIoContext()
{
    return *m_io_context;
}

boost::asio::ssl::context* HttpServer::GetSslContext()
{
    if (m_ssl_context)
        return m_ssl_context.get();
    return nullptr;
}

const std::chrono::milliseconds& HttpServer::GetReadTimeoutMilliseconds() const
{
    return m_read_timeout_milliseconds;
}
