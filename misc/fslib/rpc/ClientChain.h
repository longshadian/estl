#pragma once

#include <memory>
#include <deque>
#include <mutex>

#include <boost/asio.hpp>

#include "RpcEndpoint.h"

#include "MessageBuffer.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

class RpcControllerImpl;

class ClientChain
{
public: 
                            ClientChain(boost::asio::io_service& io_service, const RpcEndpoint& endpoint);
    virtual                 ~ClientChain() = default;
                            ClientChain(const ClientChain&) = delete;
    ClientChain&               operator=(const ClientChain&) = delete;

    void                    asyncConnect();
    void                    addController(int seq_id, std::shared_ptr<RpcControllerImpl> cntl);
    void                    asyncWrite(MessageBuffer&& buffer);
protected:
    void                    asyncRead();
private:
    void                    onReadSome();
    void                    parseMessage();
    void                    asyncWriteInternal();
protected:
    boost::asio::io_service&        m_io_service;
    tcp::socket                     m_socket;
    RpcEndpoint                     m_endpoint;
    MessageBuffer                   m_send_buffer;
    MessageBuffer                   m_receive_buffer;
    std::map<int, std::shared_ptr<RpcControllerImpl>> m_cntls;

    std::mutex                      m_mtx;
    std::deque<MessageBuffer>       m_send_queue;
    bool                            m_on_writing = { false };
};

}    
}
