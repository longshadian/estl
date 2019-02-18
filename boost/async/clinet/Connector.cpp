#include "Connector.h"

#include <iostream>

namespace zylib{

Connector::Connector(boost::asio::io_service& io_service, const std::string& str_ip, short port)
    : m_io_service(io_service)
    , m_socket(io_service)
    , m_server_addr(boost::asio::ip::address::from_string(str_ip), port)
    , m_is_connected(false)
{
}

bool Connector::start()
{
    m_socket.async_connect(m_server_addr, [this](const boost::system::error_code& ec) 
        {
            if (ec) {
                m_is_connected = false;
                std::cout << ec.message() << std::endl;
                return;
            }
            std::cout << "connect ok" << std::endl;
            m_is_connected = true;
            doRead();
        });
    return m_is_connected;
}

bool Connector::isConnected() const
{
    return m_is_connected;
}

void Connector::sendMsg(const std::vector<char>& msg)
{
    m_io_service.post([this, msg]()
    {
        bool wait_write = !m_write_buffer.empty();
        m_write_buffer.push_back(std::move(msg));
        if (!wait_write) {
            doWrite();
        }
    });
}

void Connector::sendMsg(const std::string& msg)
{
    std::vector<char> buff(msg.begin(), msg.end());
    sendMsg(buff);
}

void Connector::close()
{
    if (!m_is_connected.exchange(false))
        return;
    m_io_service.post([this]()
    {
        boost::system::error_code ec;
        m_socket.close(ec);
        m_is_connected = false;
    });
}

void Connector::doRead()
{
    m_socket.async_read_some(boost::asio::buffer(m_read_buffer),
        [this](boost::system::error_code ec, size_t length) 
        {
            if (ec) {
                std::cout << "read ec:" << ec.value() << " " << ec.message() << std::endl;
                m_socket.close();
                return;
            }
            std::string s(m_read_buffer.data(), m_read_buffer.data() + length);
            std::cout << "doread :" << s << std::endl;
            doRead();
        });
}

void Connector::doWrite()
{
    boost::asio::async_write(m_socket,
        boost::asio::buffer(m_write_buffer.front().data(),
            m_write_buffer.front().size()),
        [this](boost::system::error_code ec, std::size_t /*length*/)
    {
        if (ec) {
            m_socket.close(ec);
            m_is_connected = false;
            return;
        }

        m_write_buffer.pop_front();
        if (!m_write_buffer.empty()) {
            doWrite();
        }
    });
}

}