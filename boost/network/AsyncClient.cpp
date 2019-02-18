#include "AsyncClient.h"

#include <cstdlib>
#include <deque>
#include <iostream>
#include <thread>
#include <boost/asio.hpp>

#include "Message.h"

namespace network {

AsyncClient::AsyncClient(boost::asio::io_service& io_service,
    tcp::resolver::iterator endpoint_iterator) 
    : m_io_service(io_service)
    , m_socket(io_service)
    , m_write_msgs()
{
    doConnect(endpoint_iterator);
}

AsyncClient::~AsyncClient()
{
    closeSocket();
}

void AsyncClient::write(const Message& msg)
{
    m_io_service.post([this, msg]
        {
            bool wait_write = !m_write_msgs.empty();
            m_write_msgs.push_back(std::move(msg));
            if (!wait_write) {
                doWrite();
            }
        });
}

void AsyncClient::shutdown()
{
    m_io_service.post([this]() 
        { 
            closeSocket();
        });
}

void AsyncClient::doConnect(tcp::resolver::iterator endpoint_iterator)
{
    boost::asio::async_connect(m_socket, endpoint_iterator,
        [this](boost::system::error_code ec, tcp::resolver::iterator)
        {
            if (!ec) {
                doReadHeader();
            } else {
                onConnectionError(ec);
            }
        });
}

void AsyncClient::doReadHeader()
{
    boost::asio::async_read(m_socket,
        boost::asio::buffer(m_read_head.data(), m_read_head.size()),
            [this](boost::system::error_code ec, std::size_t /*length*/)
            {
                if (ec) {
                    onSocketError(ec);
                    closeSocket();
                    return;
                }
                std::memcpy(&m_total, m_read_head.data(), 4);
                std::memcpy(&m_msg_id, m_read_head.data() + 4, 4);
                if (m_total <= 8) {
                    closeSocket();
                    return;
                }

                m_read_body.resize(m_total - 4 - 4);
                doReadBody();
            });
}

void AsyncClient::doReadBody()
{
    boost::asio::async_read(m_socket,
        boost::asio::buffer(m_read_body),
            [this](boost::system::error_code ec, std::size_t /*length*/)
            {
                if (ec) {
                    onSocketError(ec);
                    closeSocket();
                    return;
                }
                m_read_head.fill(0);
                m_read_body.clear();
                m_total = 0;
                m_msg_id = 0;
                doReadHeader();
            });
}

void AsyncClient::onSocketError(const boost::system::error_code& ec)
{
    (void)ec;
}

void AsyncClient::onConnectionError(const boost::system::error_code& ec)
{
    (void)ec;
}

void AsyncClient::doWrite()
{
    boost::asio::async_write(m_socket,
        boost::asio::buffer(m_write_msgs.front().data(),
          m_write_msgs.front().size()),
        [this](boost::system::error_code ec, std::size_t /*length*/)
        {
            if (ec) {
                onSocketError(ec);
                closeSocket();
                return;
            }

            m_write_msgs.pop_front();
            if (!m_write_msgs.empty()) {
                doWrite();
            }
        });
}

void AsyncClient::closeSocket()
{
    boost::system::error_code ec;
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    m_socket.close(ec);
}

}
