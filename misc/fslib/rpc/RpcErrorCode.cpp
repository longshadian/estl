#include "RpcErrorCode.h"

namespace fslib {
namespace grpc {

#define MAKE_CASE(name) case name: return (#name)

const char* toRpcErrorText(RpcErrorCode error_code)
{
    switch (error_code)
    {
        MAKE_CASE(RpcErrorCode::RPC_SUCCESSED);
        MAKE_CASE(RpcErrorCode::RPC_ERROR_RESOLVE_ADDRESS);
        MAKE_CASE(RpcErrorCode::RPC_ERROR_PARSE_REQUEST);
        MAKE_CASE(RpcErrorCode::RPC_ERROR_PARSE_RESPONSE);
        MAKE_CASE(RpcErrorCode::RPC_ERROR_UNKNOWN);
    }
    return "RPC_ERROR_UNDEFINED";
}

}
}