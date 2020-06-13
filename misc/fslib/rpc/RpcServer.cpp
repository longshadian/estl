#include "RpcServer.h"

#include <iostream>
#include <memory>

#include "ServerChain.h"
#include "ThreadPool.h"
#include "ServicePool.h"

namespace fslib {
namespace grpc {

RpcServer::RpcServer()
{
    m_io_service = std::make_unique<boost::asio::io_service>();
    m_work_pool = std::make_shared<ThreadPool>(2);
    m_service_pool = std::make_shared<ServicePool>();
}

RpcServer::~RpcServer()
{

}

bool RpcServer::startNetwork(std::string bind_ip, uint16_t port)
{
    if (m_run.exchange(true))
        return false;

    m_acceptor = std::make_unique<tcp::acceptor>(tcp::acceptor(*m_io_service, tcp::endpoint(address::from_string(bind_ip), port)));
    m_socket = std::make_unique<tcp::socket>(*m_io_service);
    m_endpoint = std::make_unique<tcp::endpoint>();
    asyncAccept();
    return true;
}

void RpcServer::stopNetwork()
{
    m_run.store(true);
}

std::shared_ptr<ThreadPool> RpcServer::getThreadPool()
{
    return m_work_pool;
}

bool RpcServer::regService(std::shared_ptr<::google::protobuf::Service> service)
{
    return m_service_pool->regService(service);
}

std::shared_ptr<ServicePool> RpcServer::getServicePool() const
{
    return m_service_pool;
}

void RpcServer::asyncAccept()
{
    m_acceptor->async_accept(*m_socket, *m_endpoint, [this](boost::system::error_code error)
    {
        if (!error) {
            try {
                onSocketOpen();
            } catch (const boost::system::error_code& err) {
                (void)err;
            }
		}
        asyncAccept();
    });
}

void RpcServer::onSocketOpen()
{
    auto newChain = std::make_shared<ServerChain>(*m_io_service, std::move(*m_endpoint), *this);
    m_chains.push_back(newChain);
    newChain->asyncRead();
}

}
}


