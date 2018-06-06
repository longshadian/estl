#pragma once

#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <memory>

#include <boost/asio.hpp>

#include "NetworkType.h"

namespace network {

class RWHandlerFactory;

struct ServerOption
{
    int m_max_connection{65536};    //0:unlimited
    int m_timeout_seconds{0};       //0:never timeout
};

class AsyncServer
{
    friend class RWHandler;
public:
public:
    AsyncServer(boost::asio::io_service& io_service,
        std::unique_ptr<RWHandlerFactory> handler_factory,
        short port, const ServerOption& optin);
    ~AsyncServer() = default;

    void accept();
    void stop();

    boost::asio::io_service& getIoService();
    const ServerOption& getOption() const;
private:
    void stopHandler(const RWHandlerPtr& conn);
    void handleAcceptError(const boost::system::error_code& ec);
    void stopAccept();
    RWHandlerPtr createHandler();

    static void refuseAccept(boost::asio::ip::tcp::socket socket);
private:
    std::unique_ptr<RWHandlerFactory>   m_handler_factory; 
    boost::asio::io_service&            m_io_service;
    boost::asio::ip::tcp::acceptor      m_acceptor;
    boost::asio::ip::tcp::socket        m_socket;
    std::unordered_set<RWHandlerPtr>    m_handlers;
    ServerOption                        m_option;
};

}
