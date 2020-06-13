#include "RpcClientImpl.h"

#include <map>

#include <boost/system/error_code.hpp>

#include "RpcEndpoint.h"
#include "RpcClientImpl.h"
#include "RpcControllerImpl.h"
#include "ClientChain.h"
#include "MessageHead.h"
#include "MessageBuffer.h"
#include "rpc_meta.pb.h"


namespace fslib {
namespace grpc {

using namespace boost::asio::ip;


RpcClientImpl::RpcClientImpl()
    : m_increment_seq_id(0)
    , m_io_service(std::make_shared<boost::asio::io_service>())
{

}

RpcClientImpl::~RpcClientImpl()
{

}

void RpcClientImpl::CallMethod(const ::google::protobuf::Message* request,
    ::google::protobuf::Message* response,
    std::shared_ptr<RpcControllerImpl> cntl)
{
    const auto remote_endpoint = cntl->getRemoteEndPoint();

    std::shared_ptr<ClientChain> chain;
    auto it = m_chains.find(*remote_endpoint);

    int seq_id = ++m_increment_seq_id;
    if (it != m_chains.end()) {
        chain = it->second;
    } else {
        chain = createNewChain(*remote_endpoint);
        m_chains.insert({*remote_endpoint, chain});
    }
    chain->addController(seq_id, cntl);
    cntl->setResponse(response);

    rpc_meta meta;
    meta.set_id(11);
    meta.set_name(cntl->getMethodName());

    MessageHead head;
    head.m_meta_length = meta.ByteSize();
    head.m_seq_id = seq_id;
    head.m_total_length = MESSAGE_HEAD_SIZE + head.m_meta_length + request->ByteSize();

    MessageBuffer temp_buffer(head.m_total_length);
    memcpy(temp_buffer.getWritePtr(), &head, MESSAGE_HEAD_SIZE);
    temp_buffer.writeCompleted(MESSAGE_HEAD_SIZE);

    meta.SerializeToArray(temp_buffer.getWritePtr(),head.m_meta_length);
    temp_buffer.writeCompleted(head.m_meta_length);

    if (!request->SerializeToArray(temp_buffer.getWritePtr(), request->ByteSize())) {
        //TODO
        return;
    }
    temp_buffer.writeCompleted(request->ByteSize());
    assert(temp_buffer.getRemainingSpace() == 0);

    chain->asyncWrite(std::move(temp_buffer));
}

bool RpcClientImpl::resolveAddress(std::string server_address, std::shared_ptr<RpcEndpoint> endpoint)
{
    auto pos = server_address.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    std::string address = server_address.substr(0, pos);
    std::string port = server_address.substr(pos + 1);
    return resolveAddress(address, port, endpoint);
}

bool RpcClientImpl::resolveAddress(std::string address, std::string port, std::shared_ptr<RpcEndpoint> endpoint)
{
    tcp::resolver resolver(*m_io_service);
    tcp::resolver::iterator it, end;
    boost::system::error_code er;
    it = resolver.resolve(tcp::resolver::query(address, port), er);
    if (it != end) {
        *endpoint = it->endpoint();
        return true;
    }
    return false;
}

std::shared_ptr<ClientChain> RpcClientImpl::createNewChain(const RpcEndpoint& remote_endpoint)
{
    return std::make_shared<ClientChain>(*m_io_service, remote_endpoint);
}

}
}
