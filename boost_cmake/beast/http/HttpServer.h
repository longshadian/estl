#pragma once

#include <cstdint>
#include <string>
#include <functional>
#include <unordered_map>
#include <thread>
#include <vector>

#include <boost/asio.hpp>
#include "BoostBeast.h"
#include "Utility.h"

class listener;

/*
class Sender
{
public:
    virtual ~Sender() = default;
    virtual void Launch() = 0;
};

template<typename Body>
class Sender_Impl : public Sender
{
public:
    http::response<Body> m_rsp;
    Sender_Impl(http::response<Body> rsp)
        : m_rsp(std::move(rsp))
    {
    }

    virtual ~Sender_Impl() override = default;

    virtual void Launch() override
    {

    }
};
*/

class HttpServer
{
public:
    using HttpHandler = std::function<void(http::request<http::string_body>&, http::response<http::string_body>&)>;

    struct Resource
    {
        std::string m_path;
        std::string m_method;
        HttpHandler m_handler;
    };

public:
    HttpServer();
    ~HttpServer();
    HttpServer(const HttpServer& rhs) = delete;
    HttpServer& operator=(const HttpServer& rhs) = delete;
    HttpServer(HttpServer&& rhs) = delete;
    HttpServer& operator=(HttpServer&& rhs) = delete;

    void                                InitHost(std::string host);
    void                                InitPort(uint16_t port);
    void                                InitDocRoot(std::string doc_root);
    void                                InitResource(std::vector<Resource> resource_array);
    void                                InitReadTimeoutMilliseconds(std::chrono::milliseconds millisec);
    void                                InitSslContext();
    bool                                Start(int thread_num);
    void                                Stop();
    void                                WaitExit();

    const std::string&                  GetDocRoot() const;
    boost::asio::io_context&            GetIoContext();
    boost::asio::ssl::context*          GetSslContext();
    const std::chrono::milliseconds&    GetReadTimeoutMilliseconds() const;

    template<typename Send>
    void HandleRequest(http::request<http::string_body>&& req, Send&& send);

private:
    std::string                     m_doc_root;
    std::string                     m_host;
    uint16_t                        m_port;
    std::chrono::milliseconds       m_read_timeout_milliseconds;

    using HandleMap = std::unordered_map<std::string, HttpHandler>;
    std::unordered_map<std::string, HandleMap> m_resourcesMap;

    std::vector<std::thread>        m_thread_pool;
    std::unique_ptr<boost::asio::io_context> m_io_context;
    std::unique_ptr<boost::asio::ssl::context> m_ssl_context;
    std::shared_ptr<listener>       m_listner;
};


template<typename Send>
void HttpServer::HandleRequest(http::request<http::string_body>&& req, Send&& send)
{
    std::string method{};
    if (req.method() == http::verb::get)
        method = "GET";
    else if (req.method() == http::verb::post)
        method = "POST";
    if (method.empty()) {
        return send(BadRequest(req, "Unknown HTTP-method"));
    }

    // path
    const auto path_it = m_resourcesMap.find(req.target().to_string());
    if (path_it == m_resourcesMap.end()) {
        return send(NotFound(req));
    }

    // method
    const auto& path_resource = path_it->second;
    const auto handle_it = path_resource.find(method);
    if (handle_it == path_resource.end()) {
        return send(BadRequest(req, "Unknown HTTP-method"));
    }

    http::response<http::string_body> resp{http::status::ok, req.version()};
    resp.keep_alive(req.keep_alive());
    handle_it->second(req, resp);
    resp.prepare_payload();
    send(std::move(resp));
}
