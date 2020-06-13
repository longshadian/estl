#pragma once

#include <memory>

#include <google/protobuf/service.h>

namespace fslib {
namespace grpc {

class RpcClientImpl;

class RpcClient 
{
public:
                            RpcClient();
                            ~RpcClient() = default;
                            RpcClient(const RpcClient&) = delete;
    RpcClient&              operator=(const RpcClient&) = delete;

    std::shared_ptr<RpcClientImpl> getImpl();
private:
    std::shared_ptr<RpcClientImpl> m_impl;
};

}
}
