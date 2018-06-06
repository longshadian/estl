#pragma once

#include <cstdlib>
#include <deque>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>

namespace network {

using boost::asio::ip::tcp;

class Message;

class AsyncClient
{
public:
    AsyncClient(boost::asio::io_service& io_service, tcp::resolver::iterator endpoint_iterator);
    virtual ~AsyncClient();

    virtual void doReadHeader();
    virtual void doReadBody();
    virtual void onSocketError(const boost::system::error_code& ec);
    virtual void onConnectionError(const boost::system::error_code& ec);

    void write(const Message& msg);
    void shutdown();
private:
    void doConnect(tcp::resolver::iterator endpoint_iterator);
    void doWrite();
    void closeSocket();
private:
    boost::asio::io_service& m_io_service;
    tcp::socket m_socket;
    std::deque<Message> m_write_msgs;

    std::array<uint8_t, 4+4> m_read_head;
    std::vector<uint8_t>     m_read_body;
    int32_t                  m_total;
    int32_t                  m_msg_id;
};

}
