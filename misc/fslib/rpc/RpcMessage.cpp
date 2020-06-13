#include "RpcMessage.h"

namespace fslib {
namespace grpc {

RpcMessage::RpcMessage(RpcMessage&& rhs)
{
    m_seq_id = rhs.m_seq_id;
    m_service_name = std::move(rhs.m_service_name);
    m_data = std::move(rhs.m_data);
}
RpcMessage& RpcMessage::operator=(RpcMessage&& rhs)
{
    m_seq_id = rhs.m_seq_id;
    m_service_name = std::move(rhs.m_service_name);
    m_data = std::move(rhs.m_data);
    return *this;
}

void RpcMessage::setSeqId(uint32_t seq_id)
{
    m_seq_id = seq_id;
}

uint32_t RpcMessage::getSeqId() const
{
    return m_seq_id;
}

void RpcMessage::setServiceName(std::string service_name)
{
    m_service_name = service_name;
}

std::string RpcMessage::getServiceName() const
{
    return m_service_name;
}

void RpcMessage::setMessageBuffer(const uint8_t* p, size_t length)
{
    m_data.assign(p, length);
}

MessageBuffer RpcMessage::getMessageBuffer() const
{
    return m_data;
}

}
}