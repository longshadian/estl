#pragma once

#include <string>
#include <atomic>
#include <thread>
#include <deque>

#include <boost/asio.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_service.hpp>
#include <google/protobuf/service.h>

#include "RpcEndpoint.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

class ServerChain;
class RpcMessage;
class ThreadPool;
class ServicePool;

class RpcServer
{
public:
                            RpcServer();
                            ~RpcServer();

    bool                    startNetwork(std::string bind_ip, uint16_t port);
    void                    stopNetwork();
    std::shared_ptr<ThreadPool> getThreadPool();

    bool                    regService(std::shared_ptr<::google::protobuf::Service> service);
    std::shared_ptr<ServicePool> getServicePool() const;
protected:
    void                    onSocketOpen();
    void                    asyncAccept();
protected:
    std::unique_ptr<boost::asio::io_service>        m_io_service;
    std::atomic<bool>                               m_run = { false };
	std::unique_ptr<tcp::acceptor>                  m_acceptor;
    std::unique_ptr<tcp::socket>                    m_socket;
    std::unique_ptr<RpcEndpoint>                    m_endpoint;

    typedef std::vector<std::shared_ptr<ServerChain>> ChainVector;
    ChainVector                                     m_chains;

    std::shared_ptr<ThreadPool>                     m_work_pool;
    std::shared_ptr<ServicePool>                    m_service_pool;
};

}
}
