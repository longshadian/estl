#include "RpcChannel.h"

#include "RpcChannelImpl.h"
#include "RpcClient.h"

namespace fslib {
namespace grpc {

RpcChannel::RpcChannel(std::string server_address)
    : m_impl(std::make_shared<RpcChannelImpl>(server_address, m_client->getImpl()))
{
    m_impl->init();
}

void RpcChannel::CallMethod(const ::google::protobuf::MethodDescriptor* method,
    ::google::protobuf::RpcController* controller,
    const ::google::protobuf::Message* request,
    ::google::protobuf::Message* response,
    ::google::protobuf::Closure* done)
{
    m_impl->CallMethod(method, controller, request, response, done);
}


}
}

