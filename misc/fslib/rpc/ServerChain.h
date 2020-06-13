#pragma once

#include <memory>
#include <mutex>
#include <boost/asio.hpp>

#include "RpcEndpoint.h"
#include "RpcServer.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

class RpcControllerImpl;
class RpcServer;

class ServerChain : public std::enable_shared_from_this<ServerChain>
{
public: 
                            ServerChain(boost::asio::io_service& io_service, const RpcEndpoint& endpoint, RpcServer& server);
    virtual                 ~ServerChain() = default;
                            ServerChain(const ServerChain&) = delete;
    ServerChain&            operator=(const ServerChain&) = delete;

    void                    addController(int seq_id, std::shared_ptr<RpcControllerImpl> cntl);
    void                    asyncWrite(MessageBuffer&& buffer);
    void                    asyncRead();
    const RpcServer&        getRpcServer() const;
private:
    void                    onReadSome();
    void                    parseMessage();
    void                    asyncWriteInternal();
private:
    boost::asio::io_service&        m_io_service;
    tcp::socket                     m_socket;
    RpcEndpoint                     m_endpoint;
    MessageBuffer                   m_send_buffer;
    MessageBuffer                   m_receive_buffer;

    RpcServer&                      m_server;
    std::mutex                      m_mtx;
    std::deque<MessageBuffer>       m_send_queue;
    bool                            m_on_writing = { false };
};

}    
}
