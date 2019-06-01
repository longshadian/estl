#pragma once

#include "BoostBeast.h"
#include "Utility.h"

#include "ssl_stream.hpp"
#include "HttpServer.h"

// Echoes back all received WebSocket messages.
// This uses the Curiously Recurring Template Pattern so that
// the same code works with both SSL streams and regular sockets.
template<class Derived>
class websocket_session
{
    // Access the derived class, this is part of
    // the Curiously Recurring Template Pattern idiom.
    Derived& derived()
    {
        return static_cast<Derived&>(*this);
    }

public:
    // Construct the session
    explicit
    websocket_session(HttpServer& http_server, boost::asio::io_context& ioc)
        : http_server_(http_server)
        , strand_(ioc.get_executor())
        , timer_(ioc,
            (std::chrono::steady_clock::time_point::max)())
    {
    }

    // Start the asynchronous operation
    template<class Body, class Allocator>
    void do_accept(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
    {
        // Set the timer
        timer_.expires_after(std::chrono::seconds(15));

        // Accept the websocket handshake
        derived().ws().async_accept(
            req,
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &websocket_session::on_accept,
                    derived().shared_from_this(),
                    std::placeholders::_1)));
    }

    // Called when the timer expires.
    void on_timer(boost::system::error_code ec)
    {
        if(ec && ec != boost::asio::error::operation_aborted)
            return fail(ec, "timer");

        // Verify that the timer really expired since the deadline may have moved.
        if(timer_.expiry() <= std::chrono::steady_clock::now())
            derived().do_timeout();

        // Wait on the timer
        timer_.async_wait(
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &websocket_session::on_timer,
                    derived().shared_from_this(),
                    std::placeholders::_1)));
    }

    void on_accept(boost::system::error_code ec)
    {
        // Happens when the timer closes the socket
        if(ec == boost::asio::error::operation_aborted)
            return;

        if(ec)
            return fail(ec, "accept");

        // Read a message
        do_read();
    }

    void do_read()
    {
        // Set the timer
        timer_.expires_after(std::chrono::seconds(15));

        // Read a message into our buffer
        derived().ws().async_read(
            buffer_,
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &websocket_session::on_read,
                    derived().shared_from_this(),
                    std::placeholders::_1,
                    std::placeholders::_2)));
    }

    void on_read(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // Happens when the timer closes the socket
        if(ec == boost::asio::error::operation_aborted)
            return;

        // This indicates that the websocket_session was closed
        if(ec == websocket::error::closed)
            return;

        if(ec)
            fail(ec, "read");

        // Echo the message
        derived().ws().text(derived().ws().got_text());
        derived().ws().async_write(
            buffer_.data(),
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &websocket_session::on_write,
                    derived().shared_from_this(),
                    std::placeholders::_1,
                    std::placeholders::_2)));
    }

    void on_write(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
    {
        boost::ignore_unused(bytes_transferred);

        // Happens when the timer closes the socket
        if(ec == boost::asio::error::operation_aborted)
            return;

        if(ec)
            return fail(ec, "write");

        // Clear the buffer
        buffer_.consume(buffer_.size());

        // Do another read
        do_read();
    }

protected:
    HttpServer& http_server_;
    boost::beast::multi_buffer buffer_;
    boost::asio::strand<
        boost::asio::io_context::executor_type> strand_;
    boost::asio::steady_timer timer_;
};

// Handles a plain WebSocket connection
class plain_websocket_session
    : public websocket_session<plain_websocket_session>
    , public std::enable_shared_from_this<plain_websocket_session>
{
public:
    // Create the session
    plain_websocket_session(HttpServer& http_server, boost::asio::ip::tcp::socket socket);
    ~plain_websocket_session() = default;
    plain_websocket_session(const plain_websocket_session&) = delete;
    plain_websocket_session& operator=(const plain_websocket_session&) = delete;
    plain_websocket_session(plain_websocket_session&&) = delete;
    plain_websocket_session& operator=(plain_websocket_session&&) = delete;

    // Called by the base class
    boost::beast::websocket::stream<boost::asio::ip::tcp::socket>& ws();

    // Start the asynchronous operation
    template<class Body, class Allocator>
    void run(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
    {
        // Run the timer. The timer is operated
        // continuously, this simplifies the code.
        on_timer({});

        // Accept the WebSocket upgrade request
        do_accept(std::move(req));
    }

    void do_timeout();
    void on_close(boost::system::error_code ec);

private:
    boost::beast::websocket::stream<boost::asio::ip::tcp::socket> ws_;
    bool close_ = false;
};

// Handles an SSL WebSocket connection
class ssl_websocket_session
    : public websocket_session<ssl_websocket_session>
    , public std::enable_shared_from_this<ssl_websocket_session>
{
public:
    // Create the http_session
    ssl_websocket_session(HttpServer& http_server, ssl_stream<boost::asio::ip::tcp::socket> stream);
    ~ssl_websocket_session() = default;
    ssl_websocket_session(const ssl_websocket_session&) = delete;
    ssl_websocket_session& operator=(const ssl_websocket_session&) = delete;
    ssl_websocket_session(ssl_websocket_session&&) = delete;
    ssl_websocket_session& operator=(ssl_websocket_session&&) = delete;

    // Called by the base class
    boost::beast::websocket::stream<ssl_stream<boost::asio::ip::tcp::socket>>& ws();

    // Start the asynchronous operation
    template<class Body, class Allocator>
    void run(boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
    {
        // Run the timer. The timer is operated
        // continuously, this simplifies the code.
        on_timer({});

        // Accept the WebSocket upgrade request
        do_accept(std::move(req));
    }

    void do_eof();
    void on_shutdown(boost::system::error_code ec);
    void do_timeout();

private:
    boost::beast::websocket::stream<ssl_stream<tcp::socket>> ws_;
    boost::asio::strand<
        boost::asio::io_context::executor_type> strand_;
    bool eof_ = false;
};

template<class Body, class Allocator>
void make_websocket_session(HttpServer& http_server,
    boost::asio::ip::tcp::socket socket,
    boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
{
    std::make_shared<plain_websocket_session>(http_server, std::move(socket))->run(std::move(req));
}

template<class Body, class Allocator>
void
make_websocket_session(HttpServer& http_server, ssl_stream<boost::asio::ip::tcp::socket> stream,
    boost::beast::http::request<Body, boost::beast::http::basic_fields<Allocator>> req)
{
    std::make_shared<ssl_websocket_session>(http_server, std::move(stream))->run(std::move(req));
}

