#pragma once

#include <string>
#include <cstdint>

#include "MessageBuffer.h"

namespace fslib {
namespace grpc {

class MessageBuffer;

class RpcMessage {
public:
                        RpcMessage() = default;
                        ~RpcMessage() = default;
                        RpcMessage(const RpcMessage& rhs) = default;
                        RpcMessage(RpcMessage&& rhs);
    RpcMessage&         operator=(const RpcMessage& rhs) = default;
    RpcMessage&         operator=(RpcMessage&& rhs);

    void                setSeqId(uint32_t seq_id);
    uint32_t            getSeqId() const;

    void                setServiceName(std::string service_name);
    std::string         getServiceName() const;

    void                setMessageBuffer(const uint8_t* p, size_t length);
    MessageBuffer       getMessageBuffer() const;
private:
    uint32_t            m_seq_id        = { 0 };   
    std::string         m_service_name  = {};
    MessageBuffer       m_data          = {};
};

}
}