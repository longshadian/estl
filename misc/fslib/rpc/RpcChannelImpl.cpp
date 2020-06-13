#include "RpcChannelImpl.h"

#include <boost/log/trivial.hpp>

#include <google/protobuf/descriptor.h>

#include "RpcController.h"
#include "RpcControllerImpl.h"
#include "RpcErrorCode.h"
#include "RpcClientImpl.h"
#include "rpc_meta.pb.h"

namespace fslib {
namespace grpc {


RpcChannelImpl::RpcChannelImpl(std::string server_address, std::shared_ptr<RpcClientImpl> client_impl)
    : m_server_address(server_address)
    , m_resolve_successed(true)
    , m_client_impl(client_impl)
    , m_server_endpoint()
{

}

void RpcChannelImpl::CallMethod(const ::google::protobuf::MethodDescriptor* method,
    ::google::protobuf::RpcController* controller,
    const ::google::protobuf::Message* request,
    ::google::protobuf::Message* response,
    ::google::protobuf::Closure* done)
{
    BOOST_LOG_TRIVIAL(debug) << "RpcChannelImpl::CallMethod " << method->full_name();

    auto cntl = dynamic_cast<RpcController*>(controller);
    assert(cntl != nullptr);
    auto cntl_impl = cntl->getImpl();

    if (!m_resolve_successed) {
        cntl_impl->setErrorCode(RpcErrorCode::RPC_ERROR_RESOLVE_ADDRESS);
        return;
    }

    cntl_impl->setMethodName(method->full_name());
    m_client_impl->CallMethod(request, response, cntl_impl);
    cntl_impl->wait(done);
}

bool RpcChannelImpl::init()
{
    m_resolve_successed = m_client_impl->resolveAddress(m_server_address, m_server_endpoint);
    return m_resolve_successed;
}

}
}