#pragma once

#include <string>
#include <array>
#include <atomic>
#include <list>
#include <boost/asio.hpp>

namespace zylib {

class RWHandler;

class Connector
{
    static const int READ_BUFFER_LENGTH = 1024;
public:
    Connector(boost::asio::io_service& ios, const std::string& str_ip, short port);
    ~Connector() = default;

    bool start();
    bool isConnected() const;
    void sendMsg(const std::vector<char>& msg);
    void sendMsg(const std::string& msg);
    void close();
private:
    void doRead();
    void doWrite();
private:
    boost::asio::io_service&        m_io_service;
    boost::asio::ip::tcp::socket    m_socket;
    boost::asio::ip::tcp::endpoint  m_server_addr;
    std::atomic<bool>               m_is_connected;
    std::array<char, READ_BUFFER_LENGTH> m_read_buffer;
    std::list<std::vector<char>>    m_write_buffer;
};

}
