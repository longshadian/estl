#include "RpcChain.h"

#include <boost/log/trivial.hpp>

#include "RpcControllerImpl.h"
#include "MessageHead.h"
#include "RpcErrorCode.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

RpcChain::RpcChain(boost::asio::io_service& io_service, const RpcEndpoint& endpoint)
    : m_io_service(io_service)
    , m_socket(io_service)
    , m_endpoint(endpoint)
{
}

void RpcChain::asyncConnect()
{
    m_socket.async_connect(m_endpoint, [this] (boost::system::error_code ec)
        {
            if (!ec) {
                asyncRead();
            } else {
                //TODO close??
            }
        });
}

void RpcChain::addController(int seq_id, std::shared_ptr<RpcControllerImpl> cntl)
{
    auto it = m_cntls.find(seq_id);
    if (it == m_cntls.end()) {
        m_cntls.insert({seq_id, cntl});
    } else {
        //TODO ??
    }
}

void RpcChain::asyncRead()
{
    m_socket.async_read_some(boost::asio::buffer(m_receive_buffer.getWritePtr(), m_receive_buffer.getRemainingSpace()),
        [this](boost::system::error_code ec, size_t length) 
        {
            if (!ec) {
                onReadSome();
                asyncRead();
            }
        });
}

void RpcChain::asyncWrite(std::shared_ptr<RpcControllerImpl> cntl)
{
    const auto& send_buffer = cntl->getSendBuffer();
    m_socket.async_write_some(boost::asio::buffer(send_buffer.getReadPtr(), send_buffer.getActiveSize()),
        [cntl] (boost::system::error_code ec, size_t length) 
        {
            if (!ec) {
                cntl->getSendBuffer().readCompleted(length);
            } else {
                cntl->setErrorCode(RpcErrorCode::RPC_ERROR_SEND_REQUEST);
            }
        });
}

void RpcChain::onReadSome()
{
    if (m_receive_buffer.getActiveSize() < MESSAGE_HEAD_SIZE)
        return;

    MessageHead head;
    const uint8_t* ptr = m_receive_buffer.getReadPtr();
    size_t step = 0;

    step = sizeof(head.m_total_length);
    std::memcpy(&head.m_total_length, ptr, step);
    ptr += step;
    
    step = sizeof(head.m_meta_length);
    std::memcpy(&head.m_meta_length, ptr, step);
    ptr += step;

    step = sizeof(head.m_seq_id);
    std::memcpy(&head.m_seq_id, ptr, step);
    m_receive_buffer.readCompleted(ptr - m_receive_buffer.getReadPtr());

    auto it = m_cntls.find(head.m_seq_id);
    if (it == m_cntls.end()) {
        BOOST_LOG_TRIVIAL(error) << __LINE__ << ":RpcChain::onReadSome onReadSome " << head.m_seq_id;
        return;
    }

    auto cntl = it->second;
    cntl->appendReceiveData(m_receive_buffer.getReadPtr(), m_receive_buffer.getActiveSize());
    cntl->done();
}

}

}