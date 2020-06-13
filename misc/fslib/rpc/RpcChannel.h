#pragma once

#include <memory>

#include <google/protobuf/service.h>

namespace fslib {
namespace grpc {

class RpcChannelImpl;
class RpcClient;

class RpcChannel : public ::google::protobuf::RpcChannel
{
public:
                            RpcChannel(std::string server_address);
    virtual                 ~RpcChannel() = default;
                            RpcChannel(const RpcChannel&) = delete;
    RpcChannel&             operator=(const RpcChannel&) = delete;

    virtual void            CallMethod(const ::google::protobuf::MethodDescriptor* method, ::google::protobuf::RpcController* controller, const ::google::protobuf::Message* request, ::google::protobuf::Message* response, ::google::protobuf::Closure* done) override;
private:
    std::shared_ptr<RpcChannelImpl> m_impl;
    std::shared_ptr<RpcClient>      m_client;
};

}
}
