#include "Server.h"

namespace zylib {

Server::Server(boost::asio::io_service& io_service, short port)
    : m_io_service(io_service)
    , m_acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port))
    , m_socket(io_service)
    , m_handlers()
    , m_conn_id_pools()
{
    m_conn_id_pools.resize(MAX_CONNECTION_NUM);
    std::iota(m_conn_id_pools.begin(), m_conn_id_pools.end(), 1);
}

void Server::accept()
{
    std::cout << "start listening" << std::endl;
    m_acceptor.async_accept(m_socket, [this](const boost::system::error_code& ec)
        {
            if (!ec) {
                auto handler = createHandler();
                if (handler) {
                    m_handlers.insert({handler->getConnID(), handler});
                    std::cout << "current connect count: " << m_handlers.size() << " conn_id: " <<  handler->getConnID() << std::endl;
                    handler->start();
                } else {
                    std::cout << "too much connection" << m_handlers.size() << std::endl;
                    refuseAccept(std::move(m_socket));
                }
            }
            accept();
        });
}

void Server::handleAcceptError(const boost::system::error_code& ec)
{
    std::cout << "ERROR error reason:" << ec.value() << " "  << ec.message() << std::endl;
    stopAccept();
}

void Server::refuseAccept(boost::asio::ip::tcp::socket socket)
{
    socket.close();
}

void Server::stopAccept()
{
    boost::system::error_code ec;
    m_acceptor.cancel(ec);
    m_acceptor.close(ec);
    //m_ios.stop();
}

std::shared_ptr<RWHandler> Server::createHandler()
{
    if (m_conn_id_pools.empty())
        return nullptr;
    int conn_id = m_conn_id_pools.front();
    m_conn_id_pools.pop_front();
    auto handler = std::make_shared<RWHandler>(m_io_service, std::move(m_socket));
    handler->setConnID(conn_id);
    handler->setCallBackError([this](int conn_id) 
        {
            reuseConnID(conn_id);
        });
    return handler;
}

void Server::reuseConnID(int conn_id)
{
    auto it = m_handlers.find(conn_id);
    if (it != m_handlers.end()) {
        m_handlers.erase(it);
    }
    std::cout << "current connect count:" << m_handlers.size() << std::endl;
    m_conn_id_pools.push_back(conn_id);
}


}
