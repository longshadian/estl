#pragma once

#include <google/protobuf/service.h>

namespace fslib {
namespace grpc {

class RpcServiceImpl : public ::google::protobuf::Service
{
public:
    RpcServiceImpl();
    virtual ~RpcServiceImpl() = default;
    RpcServiceImpl(const RpcServiceImpl&) = delete;
    RpcServiceImpl& operator=(const RpcServiceImpl&) = delete;

    //virtual const ::google::protobuf::ServiceDescriptor* GetDescriptor() override;

    virtual void CallMethod(const ::google::protobuf::MethodDescriptor* method,
        ::google::protobuf::RpcController* controller,
        const ::google::protobuf::Message* request,
        ::google::protobuf::Message* response,
        ::google::protobuf::Closure* done) override;

    //virtual const ::google::protobuf::Message& GetRequestPrototype(const ::google::protobuf::MethodDescriptor* method) const override;

    //virtual const ::google::protobuf::Message& GetResponsePrototype(const ::google::protobuf::MethodDescriptor* method) const override;
};

}
}
