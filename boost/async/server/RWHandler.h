#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <list>

#include <boost/asio.hpp>

namespace zylib {

class RWHandler : public std::enable_shared_from_this<RWHandler>
{
    static const int MAX_BUFFER_LENGTH = 1024;
public:
    RWHandler(boost::asio::io_service& io_service, boost::asio::ip::tcp::socket socket);
    ~RWHandler() = default;

    void start();
    void sendMsg(const std::vector<char>& msg);
    boost::asio::ip::tcp::socket& getSocket();
    void closeSocket();
    void setConnID(int id);
    int getConnID() const;

    template <typename F>
    void setCallBackError(F f)
    {
        m_callback_error = std::move(f);
    }
private:
    void handleError(boost::system::error_code ec);
    void doRead();
    void doWrite();
private:
    int                                 m_conn_id;
    boost::asio::io_service&            m_io_service;
    boost::asio::ip::tcp::socket        m_socket;
    std::function<void(int)>            m_callback_error;
    std::array<char, MAX_BUFFER_LENGTH> m_read_buffer;
    std::list<std::vector<char>>        m_write_buffer;
};

}
