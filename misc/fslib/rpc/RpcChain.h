#pragma once

#include <memory>

#include <boost/asio.hpp>

#include "RpcEndpoint.h"

#include "MessageBuffer.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

class RpcControllerImpl;

class RpcChain
{
    enum class ChainState
    {
    };
public: 
                            RpcChain(boost::asio::io_service& io_service, const RpcEndpoint& endpoint);
    virtual                 ~RpcChain() = default;
                            RpcChain(const RpcChain&) = delete;
    RpcChain&               operator=(const RpcChain&) = delete;

    void                    asyncConnect();
    void                    addController(int seq_id, std::shared_ptr<RpcControllerImpl> cntl);
    void                    asyncWrite(std::shared_ptr<RpcControllerImpl> cntl);
protected:
    void                    asyncRead();

    void                    onReadSome();
protected:
    boost::asio::io_service&        m_io_service;
    tcp::socket                     m_socket;
    RpcEndpoint                     m_endpoint;
    MessageBuffer                   m_send_buffer;
    MessageBuffer                   m_receive_buffer;
    std::map<int, std::shared_ptr<RpcControllerImpl>> m_cntls;
};

}    
}
