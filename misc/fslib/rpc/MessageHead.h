#pragma once

#include <cstdint>

namespace fslib {
namespace grpc {

struct MessageHead
{
    uint32_t m_total_length = { 0 };
    uint32_t m_meta_length = { 0 };
    uint32_t m_seq_id = { 0 };
};

enum { MESSAGE_HEAD_SIZE = sizeof(MessageHead) };

}
}
