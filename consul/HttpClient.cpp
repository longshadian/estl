#include "HttpClient.h"

using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;

HttpClient::HttpClient()
    : m_ioc()
    , m_resolver(m_ioc)
    , m_socket(m_ioc)
{
}

HttpClient::~HttpClient()
{
    Reset();
}

http::response<http::string_body> HttpClient::Request(
    http::verb method
    , const char* host, unsigned short port
    , std::string_view path
    , http::request<http::empty_body> req)
{
    tcp::endpoint endpoint{boost::asio::ip::address::from_string(host), port};
    //const auto results = m_resolver.resolve(endpoint);
    //boost::asio::connect(m_socket, results.begin(), results.end());
    //boost::asio::connect(m_socket, endpoint);
    m_socket.connect(endpoint);

    req.method(method);
    req.target(boost::beast::string_view(path.data(), path.length()));
    req.set(http::field::host, host);
    req.set(http::field::user_agent, BOOST_BEAST_VERSION_STRING);
    req.set(http::field::version, 11);
    http::write(m_socket, req);

    // This buffer is used for reading and must be persisted
    boost::beast::flat_buffer buffer;
    http::response<http::string_body> res;

    // Receive the HTTP response
    http::read(m_socket, buffer, res);

    return res;
}

void HttpClient::Reset()
{
    // Gracefully close the socket
    boost::system::error_code ec;
    m_socket.shutdown(tcp::socket::shutdown_both, ec);
}
