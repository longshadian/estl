#pragma once

#include <string>
#include <memory>

#include <google/protobuf/service.h>

#include "RpcEndpoint.h"

namespace fslib {
namespace grpc {

class RpcClientImpl;

class RpcChannelImpl : public ::google::protobuf::RpcChannel
{
public:
                            RpcChannelImpl(std::string server_address, std::shared_ptr<RpcClientImpl> client_impl);
    virtual                 ~RpcChannelImpl() = default;
                            RpcChannelImpl(const RpcChannelImpl&) = delete;
                            RpcChannelImpl& operator=(const RpcChannelImpl&) = delete;

    virtual void            CallMethod(const ::google::protobuf::MethodDescriptor* method, ::google::protobuf::RpcController* controller, const ::google::protobuf::Message* request, ::google::protobuf::Message* response, ::google::protobuf::Closure* done);
    bool                    init();
private:
    std::string                     m_server_address;
    bool                            m_resolve_successed;
    std::shared_ptr<RpcClientImpl>  m_client_impl;
    std::shared_ptr<RpcEndpoint>    m_server_endpoint;
};

}
}
