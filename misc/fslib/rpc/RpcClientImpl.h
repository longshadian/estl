#pragma once

#include <memory>

#include <boost/asio.hpp>

#include <google/protobuf/service.h>

#include "RpcEndpoint.h"

namespace fslib {
namespace grpc {

class RpcControllerImpl;
class ClientChain;

class RpcClientImpl
{
public:
                            RpcClientImpl();
                            ~RpcClientImpl();
                            RpcClientImpl(const RpcClientImpl&) = delete;
    RpcClientImpl&          operator=(const RpcClientImpl&) = delete;
                            
    void                    CallMethod(const ::google::protobuf::Message* request, ::google::protobuf::Message* response, std::shared_ptr<RpcControllerImpl> cntl);
    bool                    resolveAddress(std::string server_address, std::shared_ptr<RpcEndpoint> endpoint);
    bool                    resolveAddress(std::string address, std::string port, std::shared_ptr<RpcEndpoint> endpoint);
private:
    std::shared_ptr<ClientChain> createNewChain(const RpcEndpoint& remote_endpoint);
private:
    int                                         m_increment_seq_id;
    std::shared_ptr<boost::asio::io_service>    m_io_service;

    typedef std::map<RpcEndpoint, std::shared_ptr<ClientChain>> ChainMap;
    ChainMap                                    m_chains;
};

}
}
