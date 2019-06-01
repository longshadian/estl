#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/version.hpp>
#include <boost/asio/bind_executor.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <boost/asio/strand.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/config.hpp>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "BoostBeast.h"
#include "Utility.h"
#include "detect_ssl.hpp"
#include "server_certificate.hpp"
#include "ssl_stream.hpp"
#include "WebsocketSession.h"

#include "HttpServer.h"

// Accepts incoming connections and launches the sessions
class listener : public std::enable_shared_from_this<listener>
{
public:
    listener(HttpServer& http_server, tcp::endpoint endpoint);
    ~listener() = default;
    listener(const listener&) = delete;
    listener& operator=(const listener&) = delete;
    listener(listener&&) = delete;
    listener& operator=(listener&&) = delete;

    // Start accepting incoming connections
    void run();
    void do_accept();
    void on_accept(boost::system::error_code ec);

private:
    HttpServer&   http_server_;
    tcp::acceptor acceptor_;
    tcp::socket socket_;
};

// Detects SSL handshakes
class detect_session : public std::enable_shared_from_this<detect_session>
{
public:
    detect_session(HttpServer& http_server, boost::asio::ip::tcp::socket socket);
    ~detect_session() = default;
    detect_session(const detect_session&) = delete;
    detect_session& operator=(const detect_session&) = delete;
    detect_session(detect_session&&) = delete;
    detect_session& operator=(detect_session&&) = delete;

    // Launch the detector
    void run();
    void on_detect(boost::system::error_code ec, boost::tribool result);

private:
    HttpServer&                 http_server_;
    tcp::socket                 socket_;
    boost::beast::flat_buffer   buffer_;
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
};

// Handles an HTTP server connection.
// This uses the Curiously Recurring Template Pattern so that
// the same code works with both SSL streams and regular sockets.
template<class Derived>
class HttpSession
{
    // Access the derived class, this is part of
    // the Curiously Recurring Template Pattern idiom.
    Derived& derived() { return static_cast<Derived&>(*this); }

    // This queue is used for HTTP pipelining.
    class queue
    {
        // Maximum number of responses we will queue
        enum { limit = 8 };

        // The type-erased, saved work item
        struct Work
        {
            virtual ~Work() = default;
            virtual void operator()() = 0;
        };

        // This holds a work item
        template<bool isRequest, class Body, class Fields>
        struct Work_Impl : Work
        {
            HttpSession& self_;
            boost::beast::http::message<isRequest, Body, Fields> msg_;

            Work_Impl(
                HttpSession& self,
                boost::beast::http::message<isRequest, Body, Fields>&& msg)
                : self_(self)
                , msg_(std::move(msg))
            {
            }

            void operator()()
            {
                boost::beast::http::async_write(
                    self_.derived().stream(),
                    msg_,
                    boost::asio::bind_executor(
                        self_.strand_,
                        std::bind(
                            &HttpSession::on_write,
                            self_.derived().shared_from_this(),
                            std::placeholders::_1,
                            msg_.need_eof())));
            }
        };

        HttpSession& self_;
        std::vector<std::unique_ptr<Work>> items_;

    public:
        explicit
        queue(HttpSession& self)
            : self_(self)
        {
            static_assert(limit > 0, "queue limit must be positive");
            items_.reserve(limit);
        }

        // Returns `true` if we have reached the queue limit
        bool is_full() const
        {
            return items_.size() >= limit;
        }

        // Called when a message finishes sending
        // Returns `true` if the caller should initiate a read
        bool on_write()
        {
            BOOST_ASSERT(! items_.empty());
            auto const was_full = is_full();
            items_.erase(items_.begin());
            if(! items_.empty())
                (*items_.front())();
            return was_full;
        }

        // Called by the HTTP handler to send a response.
        template<bool isRequest, class Body, class Fields>
        void operator()(boost::beast::http::message<isRequest, Body, Fields>&& msg)
        {
            // Allocate and store the work
            items_.emplace_back(new Work_Impl<isRequest, Body, Fields>(self_, std::move(msg)));

            // If there was no previous work, start this one
            if(items_.size() == 1)
                (*items_.front())();
        }
    };

public:
    // Construct the session
    HttpSession(HttpServer& http_server, boost::beast::flat_buffer buffer)
        : http_server_(http_server)
        , req_()
        , queue_(*this)
        , timer_(http_server.GetIoContext(), std::chrono::steady_clock::time_point::max())
        , strand_(http_server.GetIoContext().get_executor())
        , buffer_(std::move(buffer))
    {
    }

    void do_read()
    {
        // Set the timer
        timer_.expires_after(http_server_.GetReadTimeoutMilliseconds());

        // Read a request
        boost::beast::http::async_read(
            derived().stream(),
            buffer_,
            req_,
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &HttpSession::on_read,
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
            return derived().do_timeout();

        // Wait on the timer
        timer_.async_wait(
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &HttpSession::on_timer,
                    derived().shared_from_this(),
                    std::placeholders::_1)));
    }

    void on_read(boost::system::error_code ec)
    {
        // Happens when the timer closes the socket
        if(ec == boost::asio::error::operation_aborted)
            return;

        // This means they closed the connection
        if(ec == boost::beast::http::error::end_of_stream)
            return derived().do_eof();

        if(ec)
            return fail(ec, "read");

        // See if it is a WebSocket Upgrade
        if(websocket::is_upgrade(req_))
        {
            // Transfer the stream to a new WebSocket session
            return make_websocket_session(http_server_, derived().release_stream(),
                std::move(req_));
        }

        // Send the response
        http_server_.HandleRequest(std::move(req_), queue_);

        // If we aren't at the queue limit, try to pipeline another request
        if(! queue_.is_full())
            do_read();
    }

    void on_write(boost::system::error_code ec, bool close)
    {
        // Happens when the timer closes the socket
        if(ec == boost::asio::error::operation_aborted)
            return;

        if(ec)
            return fail(ec, "write");

        if(close)
        {
            // This means we should close the connection, usually because
            // the response indicated the "Connection: close" semantic.
            return derived().do_eof();
        }

        // Inform the queue that a write completed
        if(queue_.on_write())
        {
            // Read another request
            do_read();
        }
    }

protected:
    HttpServer& http_server_;
    boost::beast::http::request<boost::beast::http::string_body> req_;
    queue queue_;

    boost::asio::steady_timer timer_;
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
    boost::beast::flat_buffer buffer_;
};

// Handles a plain HTTP connection
class PlainHttpSession : public HttpSession<PlainHttpSession>
    , public std::enable_shared_from_this<PlainHttpSession>
{
public:
    // Create the http_session
    PlainHttpSession(HttpServer& http_server, boost::asio::ip::tcp::socket socket,
        boost::beast::flat_buffer buffer);
    ~PlainHttpSession() = default;
    PlainHttpSession(const PlainHttpSession&) = delete;
    PlainHttpSession& operator=(const PlainHttpSession&) = delete;
    PlainHttpSession(PlainHttpSession&&) = delete;
    PlainHttpSession& operator=(PlainHttpSession&&) = delete;

    // Called by the base class
    boost::asio::ip::tcp::socket& stream();

    // Called by the base class
    boost::asio::ip::tcp::socket release_stream();

    // Start the asynchronous operation
    void run();
    void do_eof();
    void do_timeout();
private:
    boost::asio::ip::tcp::socket socket_;
    boost::asio::strand<
        boost::asio::io_context::executor_type> strand_;
};

// Handles an SSL HTTP connection
class SslHttpSession
    : public HttpSession<SslHttpSession>
    , public std::enable_shared_from_this<SslHttpSession>
{
public:
    // Create the http_session
    SslHttpSession(HttpServer& http_server, boost::asio::ip::tcp::socket socket,
        boost::beast::flat_buffer buffer);
    ~SslHttpSession() = default;
    SslHttpSession(const SslHttpSession&) = delete;
    SslHttpSession& operator=(const SslHttpSession&) = delete;
    SslHttpSession(SslHttpSession&&) = delete;
    SslHttpSession& operator=(SslHttpSession&&) = delete;

    ssl_stream<boost::asio::ip::tcp::socket>& stream();
    ssl_stream<boost::asio::ip::tcp::socket> release_stream();
    void run();
    void on_handshake(boost::system::error_code ec, std::size_t bytes_used);
    void do_eof();
    void on_shutdown(boost::system::error_code ec);
    void do_timeout();

private:
    ssl_stream<boost::asio::ip::tcp::socket> stream_;
    boost::asio::strand<boost::asio::io_context::executor_type> strand_;
    bool eof_ = false;
};
