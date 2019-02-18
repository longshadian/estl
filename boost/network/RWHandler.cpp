#include "RWHandler.h"

#include <chrono>
#include <boost/date_time/posix_time/posix_time.hpp>

#include "FakeLog.h"
#include "AsyncServer.h"

namespace network {

RWHandlerFactory::RWHandlerFactory()
{
}

RWHandlerFactory::~RWHandlerFactory()
{
}

std::shared_ptr<RWHandler> RWHandlerFactory::create(AsyncServer& server, boost::asio::ip::tcp::socket socket)
{
    return std::make_shared<RWHandler>(server, std::move(socket));
}

RWHandler::RWHandler(AsyncServer& server, boost::asio::ip::tcp::socket socket)
    : m_server(server)
    , m_io_service(m_server.getIoService())
    , m_socket(std::move(socket))
    , m_write_buffer()
    , m_is_closed(false)
{
}

RWHandler::~RWHandler()
{
}

void RWHandler::start()
{
    handlerAccept(getConnectionHdl());
}

void RWHandler::handlerAccept(ConnectionHdl hdl) { (void)hdl; }
void RWHandler::handlerClosed(ConnectionHdl hdl) { (void)hdl; }
void RWHandler::handlerTimeout(ConnectionHdl hdl) { (void)hdl; }

void RWHandler::sendMsg(std::vector<uint8_t> msg)
{
    auto self(shared_from_this());
    m_io_service.post([this, self, msg]()
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
        [this, self](boost::system::error_code ec, std::size_t length)
        {
            (void)length;
            if (ec) {
                FAKE_LOG(ERROR) << "RWHandlerBase::doWrite error: " << ec.value() << ":" << ec.message();
                onClosed();
                return;
            }
            m_write_buffer.pop_front();
            if (!m_write_buffer.empty()) {
                doWrite();
            }
        });
}

void RWHandler::doRead()
{
}

boost::asio::ip::tcp::socket& RWHandler::getSocket()
{
    return m_socket;
}

boost::asio::io_service& RWHandler::getIoService()
{
    return m_io_service;
}

void RWHandler::closeSocket()
{
    boost::system::error_code ec;
    m_socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
    m_socket.close(ec);

    m_server.stopHandler(shared_from_this());
}

void RWHandler::onClosed(CLOSED_TYPE type)
{
    FAKE_LOG(DEBUG) << "closed type:" << int(type);
    if (m_is_closed.exchange(true))
        return;
    if (type == CLOSED_TYPE::NORMAL) {
        handlerClosed(getConnectionHdl());
    } else if (type == CLOSED_TYPE::TIMEOUT) {
        handlerTimeout(getConnectionHdl());
    }
    closeSocket();
}

void RWHandler::shutdown()
{
    if (m_is_closed)
        return;
    auto self(shared_from_this());
    m_io_service.post([this, self]() 
        {
            self->onClosed(CLOSED_TYPE::ACTIVITY);
        });
}

std::shared_ptr<boost::asio::deadline_timer> RWHandler::setTimeoutTimer(int seconds)
{
    auto timer = std::make_shared<boost::asio::deadline_timer>(m_io_service);
    timer->expires_from_now(boost::posix_time::seconds(seconds));

    auto self(shared_from_this());
    timer->async_wait([self](const boost::system::error_code& ec) {
        if (!ec) {
            self->onClosed(CLOSED_TYPE::TIMEOUT);
        }
    });
    return timer;
}

void RWHandler::timeoutCancel(std::shared_ptr<boost::asio::deadline_timer> timer)
{
    if (timer) {
        boost::system::error_code ec;
        timer->cancel(ec);
    }
}

ConnectionHdl RWHandler::getConnectionHdl()
{
    return ConnectionHdl{shared_from_this()};
}

}
