#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <list>
#include <memory>

#include <boost/asio.hpp>

#include "ByteBuffer.h"

class TcpServer;

class TcpHandler;
using TcpHdl = std::weak_ptr<TcpHandler>;

class TcpHandler : public std::enable_shared_from_this<TcpHandler>
{
    using DeadlineTimerPtr = std::shared_ptr<boost::asio::deadline_timer>;

public:
    enum class CLOSED_TYPE : int
    {
        NORMAL   = 0,    // 正常关闭
        TIMEOUT  = 1,    // 超时关闭
        ACTIVITY = 2,    // 主动关闭
    };

public:
    TcpHandler(boost::asio::ip::tcp::socket socket, TcpServer& server, int64_t conn_id
        , std::function<void(int64_t)> cb);
    ~TcpHandler();
    TcpHandler(const TcpHandler&) = delete;
    TcpHandler& operator=(const TcpHandler&) = delete;
    TcpHandler(TcpHandler&&) = delete;
    TcpHandler& operator=(TcpHandler&&) = delete;

    void                            Init();

    void                            Send(std::shared_ptr<ByteBuffer> buffer);
    boost::asio::ip::tcp::socket&   GetSocket();
    boost::asio::io_context&        GetIOContext();
    void                            Shutdown();
    TcpHdl                          GetHdl();
    int64_t                         GetConnID() const;

private:
    void                            DoClosed(CLOSED_TYPE type = CLOSED_TYPE::NORMAL);
    void                            DoWrite();
    void                            WriteCallback(boost::system::error_code ec, std::size_t length);
    void                            DoRead();
    void                            TimeoutCancel(DeadlineTimerPtr timer);
    size_t                          GetTimeoutTime() const;
    bool                            Decode();
    DeadlineTimerPtr                SetTimeoutTimer(size_t seconds);

private:
    TcpServer&                              m_tcp_server;
    boost::asio::ip::tcp::socket            m_socket;
    std::atomic<bool>                       m_is_closed;
    int64_t                                 m_conn_id;
    std::array<uint8_t, 1024>               m_read_fix_buffer;
    ByteBuffer                              m_read_total_buffer;
    std::list<std::shared_ptr<ByteBuffer>>  m_write_buffer;
    std::function<void(int64_t)>            m_server_cb;
};
