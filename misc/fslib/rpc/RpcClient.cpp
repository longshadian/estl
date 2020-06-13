#include "RpcClient.h"

#include "RpcClientImpl.h"

namespace fslib {
namespace grpc {

RpcClient::RpcClient()
    : m_impl(new RpcClientImpl())
{

}

std::shared_ptr<RpcClientImpl> RpcClient::getImpl()
{
    return m_impl;
}

}
}
