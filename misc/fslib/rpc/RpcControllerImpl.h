#pragma once

#include <memory>
#include <future>

#include <google/protobuf/service.h>

#include "RpcEndpoint.h"
#include "MessageBuffer.h"

namespace fslib {
namespace grpc {

enum class RpcErrorCode;

class RpcControllerImpl : public ::google::protobuf::RpcController
{
public:
                            RpcControllerImpl();
    virtual                 ~RpcControllerImpl() = default;
                            RpcControllerImpl(const RpcControllerImpl&) = delete;
    RpcControllerImpl&      operator=(const RpcControllerImpl&) = delete;

    virtual void            Reset() override;
    virtual bool            Failed() const override;
    virtual std::string     ErrorText() const override;
    virtual void            StartCancel() override;
    virtual void            SetFailed(const std::string& reason) override;
    virtual bool            IsCanceled() const override;
    virtual void            NotifyOnCancel(::google::protobuf::Closure* callback) override;

    void                    setRemoteEndpoint(std::shared_ptr<RpcEndpoint> endpoint);
    std::shared_ptr<RpcEndpoint> getRemoteEndPoint() const;

    void                    setErrorCode(RpcErrorCode error_code);
    RpcErrorCode            getErrorCode() const;

    void                    setMethodName(std::string name);
    std::string             getMethodName() const;
    void                    setSequenceId(int seq_id);
    int                     getSequenceId() const;

    void                    setResponse(::google::protobuf::Message* response);
    ::google::protobuf::Message* getResponse();

    void                    wait(::google::protobuf::Closure* done);
    //void                    setSendBuffer(MessageBuffer buffer);
    //MessageBuffer&          getSendBuffer();
    //const MessageBuffer&    getSendBuffer() const;

    void                    appendReceiveData(const uint8_t* ptr, size_t length);
    void                    done();
private:
    int                             m_sequence_id;
    std::string                     m_method_name;
    ::google::protobuf::Message*    m_response;
    RpcErrorCode                    m_error_code;
    std::shared_ptr<RpcEndpoint>    m_endpoint;
    MessageBuffer                   m_send_buffer;
    MessageBuffer                   m_receive_buffer;

    std::future<::google::protobuf::Message*>   m_future;
    std::promise<::google::protobuf::Message*>  m_promise;
};

}
}
