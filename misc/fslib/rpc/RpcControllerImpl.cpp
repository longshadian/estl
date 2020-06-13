#include "RpcControllerImpl.h"

#include <google/protobuf/message.h>

#include "RpcErrorCode.h"


namespace fslib {
namespace grpc {

RpcControllerImpl::RpcControllerImpl()
    : m_error_code(RpcErrorCode::RPC_ERROR_UNKNOWN)
    , m_endpoint()
{
}

void RpcControllerImpl::Reset()
{
}

bool RpcControllerImpl::Failed() const
{
    return m_error_code != RpcErrorCode::RPC_SUCCESSED;
}

std::string RpcControllerImpl::ErrorText() const
{
    return toRpcErrorText(m_error_code);
}

void RpcControllerImpl::StartCancel()
{
}

void RpcControllerImpl::SetFailed(const std::string& reason)
{

}

bool RpcControllerImpl::IsCanceled() const
{
    //TODO
    return true;
}

void RpcControllerImpl::NotifyOnCancel(::google::protobuf::Closure* callback)
{
}

void RpcControllerImpl::setRemoteEndpoint(std::shared_ptr<RpcEndpoint> endpoint)
{
    m_endpoint = endpoint;
}

std::shared_ptr<RpcEndpoint> RpcControllerImpl::getRemoteEndPoint() const
{
    return m_endpoint;
}

void RpcControllerImpl::setErrorCode(RpcErrorCode error_code)
{
    m_error_code = error_code;
    m_promise.set_value(m_response);
}

RpcErrorCode RpcControllerImpl::getErrorCode() const
{
    return m_error_code;
}

void RpcControllerImpl::setMethodName(std::string name)
{
    m_method_name = name;
}

std::string RpcControllerImpl::getMethodName() const
{
    return m_method_name;
}

void RpcControllerImpl::setSequenceId(int seq_id)
{
    m_sequence_id = seq_id;
}

int RpcControllerImpl::getSequenceId() const
{
    return m_sequence_id;
}

void RpcControllerImpl::setResponse(::google::protobuf::Message* response)
{
    m_response = response;
}

::google::protobuf::Message* RpcControllerImpl::getResponse()
{
    return m_response;
}

void RpcControllerImpl::wait(::google::protobuf::Closure* done)
{
    m_future = m_promise.get_future();
    m_future.get();
}

/*
void RpcControllerImpl::setSendBuffer(MessageBuffer buffer)
{
    m_send_buffer = buffer;
}

MessageBuffer& RpcControllerImpl::getSendBuffer()
{
    return m_send_buffer;
}

const MessageBuffer& RpcControllerImpl::getSendBuffer() const
{
    return m_send_buffer;
}
*/

void RpcControllerImpl::appendReceiveData(const uint8_t* ptr, size_t length)
{
    memcpy(m_receive_buffer.getWritePtr(), ptr, length);
    m_receive_buffer.writeCompleted(length);
}

void RpcControllerImpl::done()
{
    if (m_receive_buffer.getRemainingSpace() != 0)
        return;
    if (m_response->ParseFromArray(m_receive_buffer.getReadPtr(), m_receive_buffer.getActiveSize())) {
        setErrorCode(RpcErrorCode::RPC_ERROR_PARSE_RESPONSE);
    }
    m_receive_buffer.readCompleted(m_receive_buffer.getActiveSize());
    m_promise.set_value(m_response);
}

}
}
