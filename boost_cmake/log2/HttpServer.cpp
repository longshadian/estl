#include "HttpServer.h"

void
fail(boost::system::error_code ec, char const* what)
{
    BOOST_LOG_TRIVIAL(warning) << what << ": " << ec.message();
}

Listener::Listener(boost::asio::io_context& ioc, tcp::endpoint endpoint, const std::string& doc_root)
    : m_acceptor(ioc)
    , m_socket(ioc)
    , m_doc_root(doc_root)
{
    m_acceptor.open(endpoint.protocol());
    m_acceptor.bind(endpoint);
    m_acceptor.listen(boost::asio::socket_base::max_listen_connections);
}

void Listener::Run()
{
    if (!m_acceptor.is_open())
        return;
    DoAccept();
}

void Listener::DoAccept()
{
    m_acceptor.async_accept(m_socket, std::bind(&Listener::OnAccept, shared_from_this(), std::placeholders::_1));
}

void Listener::OnAccept(boost::system::error_code ec)
{
    if (ec) {
    } else {
    std::make_shared<http_session>(
        std::move(m_socket),
        m_doc_root)->run();
    }

    DoAccept();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

websocket_session::websocket_session(tcp::socket socket)
    : ws_(std::move(socket))
    , strand_(ws_.get_executor())
    , timer_(ws_.get_executor().context(),
    (std::chrono::steady_clock::time_point::max)())
{
}

void websocket_session::on_timer(boost::system::error_code ec)
{
    if (ec && ec != boost::asio::error::operation_aborted)
        return fail(ec, "timer");

    // Verify that the timer really expired since the deadline may have moved.
    if (timer_.expiry() <= std::chrono::steady_clock::now())
    {
        // Closing the socket cancels all outstanding operations. They
        // will complete with boost::asio::error::operation_aborted
        ws_.next_layer().shutdown(tcp::socket::shutdown_both, ec);
        ws_.next_layer().close(ec);
        return;
    }

    // Wait on the timer
    timer_.async_wait(
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &websocket_session::on_timer,
                shared_from_this(),
                std::placeholders::_1)));
}


void websocket_session::on_accept(boost::system::error_code ec)
{
    // Happens when the timer closes the socket
    if (ec == boost::asio::error::operation_aborted)
        return;

    if (ec)
        return fail(ec, "accept");

    // Read a message
    do_read();
}

void websocket_session::do_read()
{
    // Set the timer
    timer_.expires_after(std::chrono::seconds(15));

    // Read a message into our buffer
    ws_.async_read(
        buffer_,
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &websocket_session::on_read,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2)));
}

void websocket_session::on_read(boost::system::error_code ec, std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    // Happens when the timer closes the socket
    if (ec == boost::asio::error::operation_aborted)
        return;

    // This indicates that the websocket_session was closed
    if (ec == websocket::error::closed)
        return;

    if (ec)
        fail(ec, "read");

    // Echo the message
    ws_.text(ws_.got_text());
    ws_.async_write(
        buffer_.data(),
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &websocket_session::on_write,
                shared_from_this(),
                std::placeholders::_1,
                std::placeholders::_2)));
}


void websocket_session::on_write(
        boost::system::error_code ec,
        std::size_t bytes_transferred)
{
    boost::ignore_unused(bytes_transferred);

    // Happens when the timer closes the socket
    if (ec == boost::asio::error::operation_aborted)
        return;

    if (ec)
        return fail(ec, "write");

    // Clear the buffer
    buffer_.consume(buffer_.size());

    // Do another read
    do_read();
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

http_session::http_session(tcp::socket socket, std::string const& doc_root)
    : socket_(std::move(socket))
    , strand_(socket_.get_executor())
    , timer_(socket_.get_executor().context(),
    (std::chrono::steady_clock::time_point::max)())
    , doc_root_(doc_root)
    , queue_(*this)
{
}

void http_session::run()
{
    on_timer({});
    do_read();
}

void http_session::do_read()
{
    timer_.expires_after(std::chrono::seconds(15));

    boost::beast::http::async_read(socket_, buffer_, req_,
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &http_session::on_read,
                shared_from_this(),
                std::placeholders::_1)));
}

void http_session::on_timer(boost::system::error_code ec)
{
    if (ec && ec != boost::asio::error::operation_aborted)
        return fail(ec, "timer");

    // Verify that the timer really expired since the deadline may have moved.
    if (timer_.expiry() <= std::chrono::steady_clock::now())
    {
        // Closing the socket cancels all outstanding operations. They
        // will complete with boost::asio::error::operation_aborted
        socket_.shutdown(tcp::socket::shutdown_both, ec);
        socket_.close(ec);
        return;
    }

    // Wait on the timer
    timer_.async_wait(
        boost::asio::bind_executor(
            strand_,
            std::bind(
                &http_session::on_timer,
                shared_from_this(),
                std::placeholders::_1)));
}

void http_session::on_read(boost::system::error_code ec)
{
    // Happens when the timer closes the socket
    if (ec == boost::asio::error::operation_aborted)
        return;

    // This means they closed the connection
    if (ec == http::error::end_of_stream)
        return do_close();

    if (ec)
        return fail(ec, "read");

    // See if it is a WebSocket Upgrade
    if (websocket::is_upgrade(req_))
    {
        // Create a WebSocket websocket_session by transferring the socket
        std::make_shared<websocket_session>(
            std::move(socket_))->run(std::move(req_));
        return;
    }

    // Send the response
    handle_request(doc_root_, std::move(req_), queue_);

    // If we aren't at the queue limit, try to pipeline another request
    if (!queue_.is_full())
        do_read();
}

void http_session::on_write(boost::system::error_code ec, bool close)
{
    // Happens when the timer closes the socket
    if (ec == boost::asio::error::operation_aborted)
        return;

    if (ec)
        return fail(ec, "write");

    if (close)
    {
        // This means we should close the connection, usually because
        // the response indicated the "Connection: close" semantic.
        return do_close();
    }

    // Inform the queue that a write completed
    if (queue_.on_write())
    {
        // Read another request
        do_read();
    }
}

void http_session::do_close()
{
    // Send a TCP shutdown
    boost::system::error_code ec;
    socket_.shutdown(tcp::socket::shutdown_send, ec);

    // At this point the connection is closed gracefully
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////


HttpServer::HttpServer(uint16_t port)
    : m_host("127.0.0.1")
    , m_port(port)
    , m_threads()
{
}

HttpServer::HttpServer(std::string host, uint16_t port)
    : m_host(std::move(host))
    , m_port(port)
    , m_threads()
{
}

HttpServer::~HttpServer()
{
}

bool HttpServer::Initialize(int32_t threads)
{
    std::string doc_root = "/";
    try {
        m_address = boost::asio::ip::make_address(m_host);
        boost::asio::io_context ioc{threads};
        // Create and launch a listening port
        std::make_shared<Listener>(
            ioc,
            tcp::endpoint{ m_address, m_port },
            doc_root)->Run();

        // Run the I/O service on the requested number of threads
        std::vector<std::thread> v;
        v.reserve(threads - 1);
        for (auto i = threads - 1; i > 0; --i)
            v.emplace_back(
                [&ioc]
        {
            ioc.run();
        });
        ioc.run();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

void HttpServer::Stop()
{
}

int main() 
{
    HttpServer server{9999};
    server.Initialize(1);
    return 0;
}
