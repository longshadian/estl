#pragma once

#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <memory>
#include <functional>
#include <thread>
#include <atomic>

#include <boost/asio.hpp>

#include "ByteBuffer.h"

class TcpServerEvent;
class TcpHandler;

class TcpServer
{
public:
    struct Option
    {
        size_t m_max_connection{65535};    //0:unlimited
        size_t m_timeout_seconds{0};       //0:never timeout
    };

    friend class RWHandler;
public:
    TcpServer(uint16_t port, std::unique_ptr<TcpServerEvent> evt, Option option);
    ~TcpServer();
    TcpServer(const TcpServer& rhs) = delete;
    TcpServer& operator=(const TcpServer& rhs) = delete;
    TcpServer(TcpServer&& rhs) = delete;
    TcpServer& operator=(TcpServer&& rhs) = delete;

    void                            Start();
    void                            Stop();
    void                            WaitThreadExit();
    boost::asio::io_service&        GetIOContext();
    const Option&                   GetOption() const;
    TcpServerEvent&                 GetEvent();

private:
    void                            Run();
    void                            StopHandler(int64_t conn_id);
    void                            StopAccept();
    void                            StopAllHandler();
    void                            DoAccept();
    std::shared_ptr<TcpHandler>     CreateHandler();
            
private:
    std::atomic<bool>                               m_is_running;
    std::thread                                     m_thread;
    boost::asio::io_context                         m_io_context;
    std::unique_ptr<boost::asio::io_service::work>  m_work;
    boost::asio::ip::tcp::acceptor                  m_acceptor;
    boost::asio::ip::tcp::socket                    m_socket;
    std::unordered_map<int64_t, std::shared_ptr<TcpHandler>> m_handlers;
    Option                                          m_option;
    std::unique_ptr<TcpServerEvent>                 m_event;
    std::atomic<int64_t>                            m_handler_index;
};
