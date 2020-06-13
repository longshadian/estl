#pragma once

#include <memory>

#include <google/protobuf/service.h>

namespace fslib {
namespace grpc {

class RpcControllerImpl;

class RpcController : public ::google::protobuf::RpcController
{
public:
                            RpcController();
    virtual                 ~RpcController() = default;
                            RpcController(const RpcController&) = delete;
    RpcController&          operator=(const RpcController&) = delete;

    virtual void            Reset() override;
    virtual bool            Failed() const override;
    virtual std::string     ErrorText() const override;
    virtual void            StartCancel() override;
    virtual void            SetFailed(const std::string& reason) override;
    virtual bool            IsCanceled() const override;
    virtual void            NotifyOnCancel(::google::protobuf::Closure* callback) override;

    std::shared_ptr<RpcControllerImpl> getImpl();
private:
    std::shared_ptr<RpcControllerImpl> m_impl;
};

}
}
