#pragma once

#include <string>

namespace fslib {
namespace grpc {

enum class RpcErrorCode
{
    RPC_SUCCESSED                   = 0,
    RPC_ERROR_RESOLVE_ADDRESS       = 1,
    RPC_ERROR_PARSE_REQUEST         = 2,
    RPC_ERROR_PARSE_RESPONSE        = 3,
    RPC_ERROR_SEND_REQUEST          = 4,
    RPC_ERROR_RECEIVE_RESPONSE      = 5,

    RPC_ERROR_UNKNOWN               = 100,
};

const char*                         toRpcErrorText(RpcErrorCode code);

}
}
