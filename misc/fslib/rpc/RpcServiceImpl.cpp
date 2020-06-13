#include "RpcServiceImpl.h"

namespace fslib {
namespace grpc {

RpcServiceImpl::RpcServiceImpl()
{
}

/*
const ::google::protobuf::ServiceDescriptor* RpcServiceImpl::GetDescriptor()
{
}
*/

void RpcServiceImpl::CallMethod(const ::google::protobuf::MethodDescriptor* method,
    ::google::protobuf::RpcController* controller,
    const ::google::protobuf::Message* request,
    ::google::protobuf::Message* response,
    ::google::protobuf::Closure* done)
{
}

/*
const ::google::protobuf::Message& RpcServiceImpl::GetRequestPrototype(const ::google::protobuf::MethodDescriptor* method) const
{
}

const ::google::protobuf::Message& RpcServiceImpl::GetResponsePrototype(const ::google::protobuf::MethodDescriptor* method) const
{
}
*/

}
}
