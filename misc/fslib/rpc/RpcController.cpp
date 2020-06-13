#include "RpcController.h"

#include "RpcControllerImpl.h"

namespace fslib {
namespace grpc {

RpcController::RpcController()
    : m_impl(new RpcControllerImpl())
{
}

void RpcController::Reset()
{
    m_impl->Reset();
}

bool RpcController::Failed() const
{
    return m_impl->Failed();
}

std::string RpcController::ErrorText() const
{
    return m_impl->ErrorText();
}

void RpcController::StartCancel()
{
    m_impl->StartCancel();
}

void RpcController::SetFailed(const std::string& reason)
{
    m_impl->SetFailed(reason);
}

bool RpcController::IsCanceled() const
{
    return m_impl->IsCanceled();
}

void RpcController::NotifyOnCancel(::google::protobuf::Closure* callback)
{
    m_impl->NotifyOnCancel(callback);
}

std::shared_ptr<RpcControllerImpl> RpcController::getImpl()
{
    return m_impl;
}

}
}
