#pragma once

#include <memory>
#include <tuple>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/service.h>

#include "RpcMessage.h"
#include "RpcErrorCode.h"

namespace fslib {
namespace grpc {

class ServerChain;
class ServiceSlot;
class RpcController;

class RpcRequest
{
public:
                            RpcRequest(std::shared_ptr<ServerChain> chain);
                            ~RpcRequest() = default;
                            RpcRequest(RpcRequest&& rhs);
    RpcRequest&             operator=(RpcRequest&& rhs);

    RpcMessage&             getRpcMessage();
    void                    callRpc();
private:
    void                    callMethod(std::shared_ptr<ServiceSlot>,
                                       const ::google::protobuf::MethodDescriptor*,
                                       std::shared_ptr<RpcController>,
                                       std::shared_ptr<::google::protobuf::Message> request,
                                       std::shared_ptr<::google::protobuf::Message> response);
    void                    onCallMethodDone(std::tuple<std::shared_ptr<RpcController>,
                                                        std::shared_ptr<::google::protobuf::Message>,
                                                        std::shared_ptr<::google::protobuf::Message>
                                                        > arguments);
    void                    sendErrorCode(RpcErrorCode code);
    static bool             parseFullName(std::string name, std::string* service_name, std::string* method_name);
private:
    RpcMessage                      m_message;
    std::shared_ptr<ServerChain>    m_chain;
};

//////////////////////////////////////////////////////////////////////////
}
}