#include "AsyncServer.h"

#include "RWHandler.h"
#include "FakeLog.h"

namespace network {

AsyncServer::AsyncServer(boost::asio::io_service& io_service,
    std::unique_ptr<RWHandlerFactory> handler_factory,
    short port, const ServerOption& option)
    : m_handler_factory(std::move(handler_factory))
    , m_io_service(io_service)
    , m_acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port))
    , m_socket(io_service)
    , m_handlers()
    , m_option(option)
{
}

void AsyncServer::accept()
{
    m_acceptor.async_accept(m_socket,
        [this](const boost::system::error_code& ec)
        {
            if (!ec) {
                auto handler = createHandler();
                if (handler) {
                    m_handlers.insert(handler);
                    handler->start();
                } else {
                    FAKE_LOG(WARNING) << "refuse accept: too much connection " << m_handlers.size() << "/" << m_option.m_max_connection;
                    refuseAccept(std::move(m_socket));
                }
                FAKE_LOG(DEBUG) << "current handler:" << m_handlers.size();
            } else {
                FAKE_LOG(ERROR) << "accept error reason:" << ec.value() << " "  << ec.message();
                handleAcceptError(ec);
                return;
            }
            accept();
        });
}

void AsyncServer::stop()
{
    stopAccept();
    m_io_service.stop();
}

void AsyncServer::handleAcceptError(const boost::system::error_code& ec)
{
    (void)ec;
    stopAccept();
}

void AsyncServer::refuseAccept(boost::asio::ip::tcp::socket socket)
{
    boost::system::error_code ec;
    socket.shutdown(boost::asio::socket_base::shutdown_both, ec);
    socket.close(ec);
}

void AsyncServer::stopAccept()
{
    boost::system::error_code ec;
    m_acceptor.cancel(ec);
    m_acceptor.close(ec);
}

std::shared_ptr<RWHandler> AsyncServer::createHandler()
{
    if (m_option.m_max_connection != 0 && (int)m_handlers.size() >= m_option.m_max_connection) {
        return nullptr;
    }
    return m_handler_factory->create(*this, std::move(m_socket));
}

void AsyncServer::stopHandler(const RWHandlerPtr& conn)
{
    m_handlers.erase(conn);
    FAKE_LOG(DEBUG) << "current connect count:" << m_handlers.size();
}

boost::asio::io_service& AsyncServer::getIoService()
{
    return m_io_service;
}

const ServerOption& AsyncServer::getOption() const
{
    return m_option;
}

}
