#include "RpcRequest.h"

#include <google/protobuf/stubs/common.h>

#include "rpc_meta.pb.h"
#include "ServerChain.h"
#include "ServicePool.h"
#include "RpcController.h"
#include "RpcErrorCode.h"
#include "RpcControllerImpl.h"
#include "MessageHead.h"

namespace fslib {
namespace grpc {

RpcRequest::RpcRequest(std::shared_ptr<ServerChain> chain)
{
    m_chain = chain;
}

RpcRequest::RpcRequest(RpcRequest&& rhs)
{
    m_message = std::move(rhs.m_message);
    m_chain = rhs.m_chain;
}

RpcRequest& RpcRequest::operator=(RpcRequest&& rhs)
{
    m_message = std::move(rhs.m_message);
    m_chain = rhs.m_chain;
    return *this;
}

RpcMessage& RpcRequest::getRpcMessage()
{
    return m_message;
}

void RpcRequest::callRpc()
{
    auto full_name = m_message.getServiceName();
    std::string service_name;
    std::string method_name;
    if (!parseFullName(full_name, &service_name, &method_name)) {
        //TODO
        return;
    }

    auto service_pool = m_chain->getRpcServer().getServicePool();
    auto service_slot = service_pool->findServiceSlot(service_name);
    if (!service_slot) {
        //TODO
        return;
    }
    auto method_descriptor = service_slot->getServiceDescriptor()->FindMethodByName(method_name);

    std::shared_ptr<::google::protobuf::Message> request;
    request.reset(service_slot->getService()->GetRequestPrototype(method_descriptor).New());
    if (!request->ParseFromArray(m_message.getMessageBuffer().getReadPtr(), m_message.getMessageBuffer().getActiveSize())) {
        //TODO
        return;
    }

    std::shared_ptr<::google::protobuf::Message> response;
    response.reset(service_slot->getService()->GetResponsePrototype(method_descriptor).New());

    std::shared_ptr<RpcController> controller = std::make_unique<RpcController>();
    auto cntl = controller->getImpl();
    cntl->setSequenceId(m_message.getSeqId());
    cntl->setMethodName(method_name);
    callMethod(service_slot, method_descriptor, controller, request, response);
}

void RpcRequest::callMethod(std::shared_ptr<ServiceSlot> service_slot,
                            const ::google::protobuf::MethodDescriptor* method_descriptor,
                            std::shared_ptr<RpcController> controller,
                            std::shared_ptr<::google::protobuf::Message> request,
                            std::shared_ptr<::google::protobuf::Message> response)
{
    std::unique_ptr<::google::protobuf::Closure> closure;
    auto arguments = std::make_tuple(controller, request, response); 
    closure.reset(::google::protobuf::NewPermanentCallback(this, &RpcRequest::onCallMethodDone, arguments));
    service_slot->getService()->CallMethod(method_descriptor, controller.get(), request.get(), response.get(), closure.get());
}

void RpcRequest::onCallMethodDone(std::tuple <  std::shared_ptr<RpcController>,
                                                std::shared_ptr<::google::protobuf::Message>,
                                                std::shared_ptr<::google::protobuf::Message>
                                             > arguments)
{
    auto controller = std::get<0>(arguments);
    auto cntl = controller->getImpl();
    //auto request = std::get<1>(arguments);
    auto response = std::get<2>(arguments);

    if (cntl->getErrorCode() != RpcErrorCode::RPC_SUCCESSED) {
        sendErrorCode(cntl->getErrorCode());
        return;
    }

    rpc_meta meta;   
    meta.set_id(cntl->getSequenceId());
    meta.set_name(cntl->getMethodName());

    MessageHead head;
    head.m_total_length = MESSAGE_HEAD_SIZE + meta.ByteSize() + response->ByteSize();
    head.m_seq_id = cntl->getSequenceId();
    head.m_meta_length = meta.ByteSize();

    MessageBuffer temp_buffer(head.m_total_length);
    std::memcpy(temp_buffer.getWritePtr(), &head, MESSAGE_HEAD_SIZE);
    temp_buffer.writeCompleted(MESSAGE_HEAD_SIZE);

    if (!meta.SerializeToArray(temp_buffer.getWritePtr(), meta.ByteSize())) {
        //TODO
        return;
    }
    temp_buffer.writeCompleted(meta.ByteSize());

    if (!response->SerializeToArray(temp_buffer.getWritePtr(), response->ByteSize())) {
        //TODO
        return;
    }

    m_chain->asyncWrite(std::move(temp_buffer));
}

void RpcRequest::sendErrorCode(RpcErrorCode code)
{
    //TODO
}

bool RpcRequest::parseFullName(std::string name, std::string* service_name, std::string* method_name)
{
    auto pos = name.rfind('.');
    if (pos == std::string::npos) {
        return false;
    }
    service_name->assign(name.begin(), name.begin()+pos);
    method_name->assign(name.begin()+pos+1, name.end());
    return true;
}

//////////////////////////////////////////////////////////////////////////
}
}