#include "HttpSession.h"

#include "HttpServer.h"


// Accepts incoming connections and launches the sessions
listener::listener(HttpServer& http_server, tcp::endpoint endpoint)
    : http_server_(http_server)
    , acceptor_(http_server_.GetIoContext())
    , socket_(http_server_.GetIoContext())
{
    boost::system::error_code ec{};

    // Open the acceptor
    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
        fail(ec, "open");
        return;
    }

    // Bind to the server address
    acceptor_.bind(endpoint, ec);
    if (ec) {
        fail(ec, "bind");
        return;
    }

    // Start listening for connections
    acceptor_.listen(boost::asio::socket_base::max_listen_connections, ec);
    if (ec) {
        fail(ec, "listen");
        return;
    }
}

// Start accepting incoming connections
void listener::run()
{
    if (!acceptor_.is_open())
        return;
    do_accept();
}

void listener::do_accept()
{
    acceptor_.async_accept(
        socket_,
        std::bind(
            &listener::on_accept,
            shared_from_this(),
            std::placeholders::_1));
}

void listener::on_accept(boost::system::error_code ec)
{
    if (ec) {
        fail(ec, "accept");
    } else {
        std::make_shared<detect_session>(http_server_, std::move(socket_))->run();
    }
    // Accept another connection
    do_accept();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

detect_session::detect_session(HttpServer& http_server, boost::asio::ip::tcp::socket socket)
    : http_server_(http_server)
    , socket_(std::move(socket))
    , buffer_()
    , strand_(socket_.get_executor())
{
}

// Launch the detector
void detect_session::run()
{
    async_detect_ssl(
        socket_,
        buffer_,
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &detect_session::on_detect,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2)));

}

void detect_session::on_detect(boost::system::error_code ec, boost::tribool result)
{
    if (ec)
        return fail(ec, "detect");

    // ÊÇssl,¶øÇÒ´æÔÚssl context
    if (result && http_server_.GetSslContext()) {
        std::make_shared<SslHttpSession>(http_server_, std::move(socket_), std::move(buffer_))->run();
        return;
    }
    std::make_shared<PlainHttpSession>(http_server_, std::move(socket_), std::move(buffer_))->run();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

PlainHttpSession::PlainHttpSession(HttpServer& http_server, boost::asio::ip::tcp::socket socket
    , boost::beast::flat_buffer buffer)
    : HttpSession<PlainHttpSession>(http_server, std::move(buffer))
    , socket_(std::move(socket))
    , strand_(socket_.get_executor())
{
}

boost::asio::ip::tcp::socket& PlainHttpSession::stream()
{
    return socket_;
}

boost::asio::ip::tcp::socket PlainHttpSession::release_stream()
{
    return std::move(socket_);
}

void PlainHttpSession::run()
{
    // Run the timer. The timer is operated
    // continuously, this simplifies the code.
    on_timer({});

    do_read();
}

void PlainHttpSession::do_eof()
{
    // Send a TCP shutdown
    boost::system::error_code ec;
    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
}

void PlainHttpSession::do_timeout()
{
    // Closing the socket cancels all outstanding operations. They
    // will complete with boost::asio::error::operation_aborted
    boost::system::error_code ec;
    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    socket_.close(ec);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

SslHttpSession::SslHttpSession(HttpServer& http_server, boost::asio::ip::tcp::socket socket
    , boost::beast::flat_buffer buffer)
    : HttpSession<SslHttpSession>(http_server, std::move(buffer))
    , stream_(std::move(socket), *http_server.GetSslContext())
    , strand_(stream_.get_executor())
{
}

ssl_stream<boost::asio::ip::tcp::socket>& SslHttpSession::stream()
{
    return stream_;
}

ssl_stream<boost::asio::ip::tcp::socket> SslHttpSession::release_stream()
{
    return std::move(stream_);
}

void SslHttpSession::run()
{
    // Run the timer. The timer is operated
    // continuously, this simplifies the code.
    on_timer({});

    // Set the timer
    timer_.expires_after(std::chrono::seconds(15));

    // Perform the SSL handshake
    // Note, this is the buffered version of the handshake.
    stream_.async_handshake(
        boost::asio::ssl::stream_base::server,
        buffer_.data(),
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &SslHttpSession::on_handshake,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2)));
}

void SslHttpSession::on_handshake(boost::system::error_code ec, std::size_t bytes_used)
{
    // Happens when the handshake times out
    if(ec == boost::asio::error::operation_aborted)
        return;

    if(ec)
        return fail(ec, "handshake");

    // Consume the portion of the buffer used by the handshake
    buffer_.consume(bytes_used);

    do_read();
}

void SslHttpSession::do_eof()
{
    eof_ = true;

    // Set the timer
    timer_.expires_after(std::chrono::seconds(15));

    // Perform the SSL shutdown
    stream_.async_shutdown(
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &SslHttpSession::on_shutdown,
                shared_from_this(),
                std::placeholders::_1)));
}

void SslHttpSession::on_shutdown(boost::system::error_code ec)
{
    // Happens when the shutdown times out
    if(ec == boost::asio::error::operation_aborted)
        return;

    if(ec)
        return fail(ec, "shutdown");

    // At this point the connection is closed gracefully
}

void SslHttpSession::do_timeout()
{
    // If this is true it means we timed out performing the shutdown
    if(eof_)
        return;

    // Start the timer again
    timer_.expires_at(
        (std::chrono::steady_clock::time_point::max)());
    on_timer({});
    do_eof();
}
