
#include "RWHandler.h"

namespace zylib {

RWHandler::RWHandler(boost::asio::io_service& io_service, boost::asio::ip::tcp::socket socket)
    : m_conn_id(0)
    , m_io_service(io_service)
    , m_socket(std::move(socket))
    , m_callback_error()
    , m_read_buffer()
{
}

void RWHandler::doRead()
{
    auto self = shared_from_this();
    m_socket.async_read_some(boost::asio::buffer(m_read_buffer),
        [this, self] (const boost::system::error_code& ec, size_t size)
        {
            if (ec) {
                handleError(ec);
                return;
            }
            std::cout << "doRead size: " << size << std::endl;

            std::vector<char> msg(m_read_buffer.data(), m_read_buffer.data() + size);
            msg.push_back('E');
            sendMsg(msg);
            doRead();
        });
}

void RWHandler::start()
{
    doRead();
}

void RWHandler::sendMsg(const std::vector<char>& msg)
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

void RWHandler::doWrite()
{
    auto self(shared_from_this());
    boost::asio::async_write(m_socket, boost::asio::buffer(m_write_buffer.front().data(), m_write_buffer.front().size()),
        [this, self](boost::system::error_code ec, std::size_t /*length*/)
        {
            if (ec) {
                handleError(ec);
                return;
            }
            m_write_buffer.pop_front();
            if (!m_write_buffer.empty()) {
                doWrite();
            }
        });
}

boost::asio::ip::tcp::socket& RWHandler::getSocket()
{
    return m_socket;
}

void RWHandler::closeSocket()
{
    auto self = shared_from_this();
    m_io_service.post([this, self]()
        {
            boost::system::error_code ec;
            m_socket.close(ec);
        });
}

void RWHandler::setConnID(int id)
{
    m_conn_id = id;
}

int RWHandler::getConnID() const
{
    return m_conn_id;
}

void RWHandler::handleError(boost::system::error_code ec)
{
    std::cout << ec.message() << std::endl;
    m_socket.close(ec);
    if (m_callback_error)
        m_callback_error(m_conn_id);
}

}
