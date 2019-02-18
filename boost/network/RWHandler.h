#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <list>

#include <boost/asio.hpp>

#include "NetworkType.h"

namespace network {

class RWHandler;
class AsyncServer;

using ConnectionHdl = std::weak_ptr<RWHandler>;

class RWHandlerFactory
{
public:
    RWHandlerFactory();
    virtual ~RWHandlerFactory();

    virtual std::shared_ptr<RWHandler> create(AsyncServer& server, boost::asio::ip::tcp::socket socket) = 0;
};

struct ConnectionInfo
{
    size_t m_timeout_seconds;
    std::string m_ip;
    std::string m_port;
};

class RWHandler : public std::enable_shared_from_this<RWHandler>
{
public:
    using ConnectionAccept = std::function<void(ConnectionHdl, const ConnectionInfo)>;
    using ConnectionClosed = std::function<void(ConnectionHdl)>;
    using ConnectionTimeout = std::function<void(ConnectionHdl)>;

    enum class CLOSED_TYPE : int
    {
        NORMAL   = 0,    //正常关闭
        TIMEOUT  = 1,    //超时关闭
        ACTIVITY = 2,    //主动关闭
    };

public:
    RWHandler(AsyncServer& async_server, boost::asio::ip::tcp::socket socket);
    virtual ~RWHandler();

    virtual void start();
    virtual void handlerAccept(ConnectionHdl hdl);
    virtual void handlerClosed(ConnectionHdl hdl);
    virtual void handlerTimeout(ConnectionHdl hdl);

    void sendMsg(std::vector<uint8_t> msg);
    boost::asio::ip::tcp::socket& getSocket();
    boost::asio::io_service& getIoService();
    void shutdown();
    void onClosed(CLOSED_TYPE type = CLOSED_TYPE::NORMAL);
protected:
    ConnectionHdl getConnectionHdl();
    void closeSocket();
    void doWrite();
    void doRead();
    std::shared_ptr<boost::asio::deadline_timer> setTimeoutTimer(int seconds);
    void timeoutCancel(std::shared_ptr<boost::asio::deadline_timer> timer);
protected:
    AsyncServer&                                m_server;
    boost::asio::io_service&                    m_io_service;
    boost::asio::ip::tcp::socket                m_socket;
    std::list<std::shared_ptr<ByteBuffer>>      m_write_buffer;
    std::atomic<bool>                           m_is_closed;
    ConnectionInfo                              m_conn_info;                      
};

}
