#pragma once

#include <memory>
#include <thread>
#include <string>
#include <vector>
    
#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast.hpp>

#include <boost/log/trivial.hpp>

using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;
namespace websocket = boost::beast::websocket;

boost::beast::string_view MIME_Type(boost::beast::string_view path)
{
    using boost::beast::iequals;
    auto const ext = [&path]
    {
        auto const pos = path.rfind(".");
        if (pos == boost::beast::string_view::npos)
            return boost::beast::string_view{};
        return path.substr(pos);
    }();
    if (iequals(ext, ".htm"))  return "text/html";
    if (iequals(ext, ".html")) return "text/html";
    if (iequals(ext, ".php"))  return "text/html";
    if (iequals(ext, ".css"))  return "text/css";
    if (iequals(ext, ".txt"))  return "text/plain";
    if (iequals(ext, ".js"))   return "application/javascript";
    if (iequals(ext, ".json")) return "application/json";
    if (iequals(ext, ".xml"))  return "application/xml";
    if (iequals(ext, ".swf"))  return "application/x-shockwave-flash";
    if (iequals(ext, ".flv"))  return "video/x-flv";
    if (iequals(ext, ".png"))  return "image/png";
    if (iequals(ext, ".jpe"))  return "image/jpeg";
    if (iequals(ext, ".jpeg")) return "image/jpeg";
    if (iequals(ext, ".jpg"))  return "image/jpeg";
    if (iequals(ext, ".gif"))  return "image/gif";
    if (iequals(ext, ".bmp"))  return "image/bmp";
    if (iequals(ext, ".ico"))  return "image/vnd.microsoft.icon";
    if (iequals(ext, ".tiff")) return "image/tiff";
    if (iequals(ext, ".tif"))  return "image/tiff";
    if (iequals(ext, ".svg"))  return "image/svg+xml";
    if (iequals(ext, ".svgz")) return "image/svg+xml";
    return "application/text";
}

// Append an HTTP rel-path to a local filesystem path.
// The returned path is normalized for the platform.
std::string
path_cat(
    boost::beast::string_view base,
    boost::beast::string_view path)
{
    if (base.empty())
        return path.to_string();
    std::string result = base.to_string();
#if BOOST_MSVC
    char constexpr path_separator = '\\';
    if (result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
    for (auto& c : result)
        if (c == '/')
            c = path_separator;
#else
    char constexpr path_separator = '/';
    if (result.back() == path_separator)
        result.resize(result.size() - 1);
    result.append(path.data(), path.size());
#endif
    return result;
}




template <class Body>
http::response<http::string_body> BadRequest(http::request<Body>& req, boost::beast::string_view why)
{
    http::response<http::string_body> res{ http::status::bad_request, req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = why.to_string();
    res.prepare_payload();
    return res;
}

template <class Body>
http::response<http::string_body> NotFound(http::request<Body>& req, boost::beast::string_view target)
{
    http::response<http::string_body> res{ http::status::not_found, req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "The resource '" + target.to_string() + "' was not found.";
    res.prepare_payload();
    return res;
};

template <class Body>
http::response<http::string_body> ServerError(http::request<Body>& req, boost::beast::string_view what)
{
    http::response<http::string_body> res{ http::status::internal_server_error, req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "An error occurred: '" + what.to_string() + "'";
    res.prepare_payload();
    return res;
};

// This function produces an HTTP response for the given
// request. The type of the response object depends on the
// contents of the request, so the interface requires the
// caller to pass a generic lambda for receiving the response.
//template <class Body, class Allocator, class Send>
template <class Body, class Send>
void handle_request(boost::beast::string_view doc_root
    //, http::request<Body, http::basic_fields<Allocator>>&& req
    , http::request<Body>&& req
    , Send&& send)
{
    // Make sure we can handle the method
    if (req.method() != http::verb::get 
        && req.method() != http::verb::head
        && req.method() != http::verb::post)
        //return send(bad_request("Unknown HTTP-method"));
        return send(BadRequest(req, "Unknown HTTP-method"));

    // Request path must be absolute and not contain "..".
    if (req.target().empty() ||
        req.target()[0] != '/' ||
        req.target().find("..") != boost::beast::string_view::npos)
        return send(BadRequest(req, "Illegal request-target"));

    if (false) {
        BOOST_LOG_TRIVIAL(debug) << "test success\n";

        http::response<http::string_body> res{http::status::ok, req.version()};
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        res.set(http::field::content_type, "text/html");
        res.keep_alive(req.keep_alive());
        res.body() = "this is from server: " + req.body();
        return send(std::move(res));
    }

    // Build the path to the requested file
    std::string path = ""; // path_cat(doc_root, req.target());
    if (req.target().back() == '/')
        path.append("index.html");

    // Attempt to open the file
    boost::beast::error_code ec;
    http::file_body::value_type body;
    body.open(path.c_str(), boost::beast::file_mode::scan, ec);

    // Handle the case where the file doesn't exist
    if (ec == boost::system::errc::no_such_file_or_directory)
        //return send(not_found(req.target()));
        return send(NotFound(req, req.target()));

    // Handle an unknown error
    if (ec)
        return send(ServerError(req, ec.message()));

    // Respond to HEAD request
    if (req.method() == http::verb::head) {
        http::response<http::empty_body> res{ http::status::ok, req.version() };
        res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
        //res.set(http::field::content_type, mime_type(path));
        res.content_length(body.size());
        res.keep_alive(req.keep_alive());
        return send(std::move(res));
    }

    // Respond to GET request
    http::response<http::file_body> res{
        std::piecewise_construct,
        std::make_tuple(std::move(body)),
        std::make_tuple(http::status::ok, req.version()) };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    //res.set(http::field::content_type, mime_type(path));
    res.content_length(body.size());
    res.keep_alive(req.keep_alive());
    return send(std::move(res));
}


class Listener : public std::enable_shared_from_this<Listener>
{
private:
    tcp::acceptor m_acceptor;
    tcp::socket m_socket;
    std::string const& m_doc_root;

public:
    Listener(boost::asio::io_context& ioc, tcp::endpoint endpoint, const std::string& doc_root);
    void Run();
    void DoAccept();
    void OnAccept(boost::system::error_code ec);
};


class websocket_session : public std::enable_shared_from_this<websocket_session>
{
    websocket::stream<tcp::socket> ws_;
    boost::asio::strand<
        boost::asio::io_context::executor_type> strand_;
    boost::asio::steady_timer timer_;
    boost::beast::multi_buffer buffer_;

public:
    // Take ownership of the socket
    explicit websocket_session(tcp::socket socket);

    // Start the asynchronous operation
    template<class Body, class Allocator>
    void run(http::request<Body, http::basic_fields<Allocator>> req)
    {
        // Run the timer. The timer is operated
        // continuously, this simplifies the code.
        on_timer({});

        // Set the timer
        timer_.expires_after(std::chrono::seconds(15));

        // Accept the websocket handshake
        ws_.async_accept(
            req,
            boost::asio::bind_executor(
                strand_,
                std::bind(
                    &websocket_session::on_accept,
                    shared_from_this(),
                    std::placeholders::_1)));
    }

    void on_timer(boost::system::error_code ec);
    void on_accept(boost::system::error_code ec);
    void do_read();
    void on_read(boost::system::error_code ec, std::size_t bytes_transferred);
    void on_write(boost::system::error_code ec, std::size_t bytes_transferred);
};

// Handles an HTTP server connection
class http_session : public std::enable_shared_from_this<http_session>
{
    // This queue is used for HTTP pipelining.
    class queue
    {
        enum
        {
            // Maximum number of responses we will queue
            limit = 8
        };

        // The type-erased, saved work item
        struct work
        {
            virtual ~work() = default;
            virtual void operator()() = 0;
        };

        http_session& self_;
        std::vector<std::unique_ptr<work>> items_;

    public:
        explicit
            queue(http_session& self)
            : self_(self)
        {
            static_assert(limit > 0, "queue limit must be positive");
            items_.reserve(limit);
        }

        // Returns `true` if we have reached the queue limit
        bool
            is_full() const
        {
            return items_.size() >= limit;
        }

        // Called when a message finishes sending
        // Returns `true` if the caller should initiate a read
        bool
            on_write()
        {
            BOOST_ASSERT(!items_.empty());
            auto const was_full = is_full();
            items_.erase(items_.begin());
            if (!items_.empty())
                (*items_.front())();
            return was_full;
        }

        template<bool isRequest, class Body, class Fields>
        struct work_impl : work
        {
            http_session& self_;
            http::message<isRequest, Body, Fields> msg_;

            work_impl(http_session& self, http::message<isRequest, Body, Fields>&& msg)
                : self_(self)
                , msg_(std::move(msg))
            {
            }

            void operator()() 
            {
                http::async_write(self_.socket_, msg_
                    , boost::asio::bind_executor(self_.strand_
                        , std::bind(&http_session::on_write
                            , self_.shared_from_this()
                            , std::placeholders::_1
                            , msg_.need_eof())));
            }
        };

        // Called by the HTTP handler to send a response.
        template<bool isRequest, class Body, class Fields>
        void operator()(http::message<isRequest, Body, Fields>&& msg)
        {
            // This holds a work item
            /*
            struct work_impl : work
            {
                http_session& self_;
                http::message<isRequest, Body, Fields> msg_;

                work_impl(
                    http_session& self,
                    http::message<isRequest, Body, Fields>&& msg)
                    : self_(self)
                    , msg_(std::move(msg))
                {
                }

                void
                    operator()()
                {
                    http::async_write(
                        self_.socket_,
                        msg_,
                        boost::asio::bind_executor(
                            self_.strand_,
                            std::bind(
                                &http_session::on_write,
                                self_.shared_from_this(),
                                std::placeholders::_1,
                                msg_.need_eof())));
                }
            };
            */

            // Allocate and store the work
            items_.emplace_back(new work_impl<isRequest, Body, Fields>(self_, std::move(msg)));

            // If there was no previous work, start this one
            if (items_.size() == 1)
                (*items_.front())();
        }
    };

    tcp::socket socket_;
    boost::asio::strand<
        boost::asio::io_context::executor_type> strand_;
    boost::asio::steady_timer timer_;
    boost::beast::flat_buffer buffer_;
    std::string const& doc_root_;
    http::request<http::string_body> req_;
    queue queue_;

public:
    explicit http_session(tcp::socket socket, std::string const& doc_root);
    void run();
    void do_read();
    void on_timer(boost::system::error_code ec);
    void on_read(boost::system::error_code ec);
    void on_write(boost::system::error_code ec, bool close);
    void do_close();
};



class HttpServer
{
public:
    HttpServer(uint16_t port);
    HttpServer(std::string host, uint16_t port);
    ~HttpServer();

    bool Initialize(int32_t threads);
    void Stop();

private:
    std::string             m_host;
    uint16_t                m_port;
    boost::asio::ip::address m_address;
    std::vector<std::thread> m_threads;
    boost::asio::io_context m_io_context;
};
