#include "WebsocketSession.h"

plain_websocket_session::plain_websocket_session(HttpServer& http_server, boost::asio::ip::tcp::socket socket)
    : websocket_session<plain_websocket_session>(http_server, socket.get_executor().context())
    , ws_(std::move(socket))
{
}

boost::beast::websocket::stream<boost::asio::ip::tcp::socket>& plain_websocket_session::ws()
{
    return ws_;
}

void plain_websocket_session::do_timeout()
{
    // This is so the close can have a timeout
    if (close_)
        return;
    close_ = true;

    // Set the timer
    timer_.expires_after(std::chrono::seconds(15));

    // Close the WebSocket Connection
    ws_.async_close(
        boost::beast::websocket::close_code::normal,
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &plain_websocket_session::on_close,
                shared_from_this(),
                std::placeholders::_1)));
}

void plain_websocket_session::on_close(boost::system::error_code ec)
{
    // Happens when close times out
    if (ec == boost::asio::error::operation_aborted)
        return;

    if (ec)
        return fail(ec, "close");

    // At this point the connection is gracefully closed
}

ssl_websocket_session::ssl_websocket_session(HttpServer& http_server, ssl_stream<boost::asio::ip::tcp::socket> stream)
    : websocket_session<ssl_websocket_session>(http_server, stream.get_executor().context())
    , ws_(std::move(stream))
    , strand_(ws_.get_executor())
{
}

boost::beast::websocket::stream<ssl_stream<boost::asio::ip::tcp::socket>>& ssl_websocket_session::ws()
{
    return ws_;
}

void ssl_websocket_session::do_eof()
{
    eof_ = true;

    // Set the timer
    timer_.expires_after(std::chrono::seconds(15));

    // Perform the SSL shutdown
    ws_.next_layer().async_shutdown(
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &ssl_websocket_session::on_shutdown,
                shared_from_this(),
                std::placeholders::_1)));
}

void ssl_websocket_session::on_shutdown(boost::system::error_code ec)
{
    // Happens when the shutdown times out
    if (ec == boost::asio::error::operation_aborted)
        return;

    if (ec)
        return fail(ec, "shutdown");

    // At this point the connection is closed gracefully
}

void ssl_websocket_session::do_timeout()
{
    // If this is true it means we timed out performing the shutdown
    if (eof_)
        return;

    // Start the timer again
    timer_.expires_at(
        (std::chrono::steady_clock::time_point::max)());
    on_timer({});
    do_eof();
}

template<class Body, class Allocator>
void make_websocket_session(
    boost::asio::ip::tcp::socket socket,
    boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
{
    std::make_shared<plain_websocket_session>(
        std::move(socket))->run(std::move(req));
}

template<class Body, class Allocator>
void
make_websocket_session(
    ssl_stream<boost::asio::ip::tcp::socket> stream,
    boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
{
    std::make_shared<ssl_websocket_session>(
        std::move(stream))->run(std::move(req));
}


