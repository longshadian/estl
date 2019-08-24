#pragma once

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
#include <unordered_map>

namespace bhttp {
using namespace boost;
using HttpHandler = std::function<void(beast::http::request<beast::http::string_body>&, beast::http::response<beast::http::string_body>&)>;
using LogFunc = std::function<void(int, const std::string&)>;


enum ELogLevel
{
    EDebug      = 0,
    EWarning = 1,
};

void SetLogFunc(LogFunc func);

void LogDebug(const std::string& content);
void LogWarning(const std::string& content);


class Listener;

// Returns a bad request response
template <typename RequestBody>
boost::beast::http::response<beast::http::string_body>
BadRequest(boost::beast::http::request<RequestBody>& req, boost::beast::string_view why)
{
    namespace http = boost::beast::http;
    boost::beast::http::response<boost::beast::http::string_body> res{ boost::beast::http::status::bad_request, req.version() };
    //res.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(boost::beast::http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = why.to_string();
    res.prepare_payload();
    return res;
}

// Returns a not found response
template <typename RequestBody>
boost::beast::http::response<beast::http::string_body>
NotFound(boost::beast::http::request<RequestBody>& req)
{
    namespace http = boost::beast::http;
    http::response<http::string_body> res{ http::status::not_found, req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "The resource '" + req.target().to_string() + "' was not found.";
    res.prepare_payload();
    return res;
};

// Returns a server error response
template <typename RequestBody>
boost::beast::http::response<beast::http::string_body>
ServerError(boost::beast::http::request<RequestBody>& req, boost::beast::string_view what)
{
    namespace http = boost::beast::http;
    http::response<http::string_body> res{ http::status::internal_server_error, req.version() };
    res.set(http::field::server, BOOST_BEAST_VERSION_STRING);
    res.set(http::field::content_type, "text/html");
    res.keep_alive(req.keep_alive());
    res.body() = "An error occurred: '" + what.to_string() + "'";
    res.prepare_payload();
    return res;
};




class HttpServer
{
    struct Resource
    {
        std::string m_path;
        std::string m_method;
        HttpHandler m_handler;
    };

public:
    HttpServer();
    ~HttpServer();

    void SetHost(std::string host);
    void SetPort(std::uint16_t port);
    void AddHttpHandler(const std::string& method, const std::string& path, HttpHandler hdl);

    bool Init(int n);
    void Shutdown();
    void Reset();
    template<typename Send>
    void HandleRequest(beast::http::request<beast::http::string_body>&& req, Send&& send) const;

private:


private:
    using HandleMap = std::unordered_map<std::string, HttpHandler>;
    std::unordered_map<std::string, HandleMap> m_resourcesMap;
    std::string                     m_host;
    std::uint16_t                   m_port;
    std::shared_ptr<boost::asio::io_context> m_ioc;

    using WorkGuard = boost::asio::executor_work_guard<boost::asio::io_context::executor_type>;
    std::shared_ptr<WorkGuard>      m_work_guard;

    std::shared_ptr<Listener>       m_listener;
    std::vector<std::thread>        m_thread_pool;
};


template<typename Send>
void HttpServer::HandleRequest(beast::http::request<beast::http::string_body>&& req, Send&& send) const
{
    std::string method{};
    if (req.method() == beast::http::verb::get)
        method = "GET";
    else if (req.method() == beast::http::verb::post)
        method = "POST";
    if (method.empty()) {
        return send(BadRequest(req, "Unknown HTTP-method"));
    }

    auto it = m_resourcesMap.find(method);
    if (it == m_resourcesMap.end()) {
        return send(NotFound(req));
    }
    const std::unordered_map<std::string, HttpHandler>& mp = it->second;;

    std::string path = req.target().to_string();
    auto it2 = mp.find(path);
    if (it2 == mp.end()) {
        return send(NotFound(req));
    }

    beast::http::response<beast::http::string_body> resp{ beast::http::status::ok, req.version() };
    resp.keep_alive(req.keep_alive());

    // TODO catch exception
    it2->second(req, resp);
    resp.prepare_payload();
    send(std::move(resp));
}

} // namespace bhttp

