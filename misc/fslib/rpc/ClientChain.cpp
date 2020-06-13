#include "ClientChain.h"

#include <boost/log/trivial.hpp>

#include "RpcControllerImpl.h"
#include "MessageHead.h"
#include "RpcErrorCode.h"

namespace fslib {
namespace grpc {

using namespace boost::asio::ip;

ClientChain::ClientChain(boost::asio::io_service& io_service, const RpcEndpoint& endpoint)
    : m_io_service(io_service)
    , m_socket(io_service)
    , m_endpoint(endpoint)
{
}

void ClientChain::asyncConnect()
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

void ClientChain::addController(int seq_id, std::shared_ptr<RpcControllerImpl> cntl)
{
    auto it = m_cntls.find(seq_id);
    if (it == m_cntls.end()) {
        m_cntls.insert({seq_id, cntl});
    } else {
        //TODO ??
    }
}

void ClientChain::asyncWrite(MessageBuffer&& buffer)
{
    if (buffer.getActiveSize() == 0)
        return;
    std::lock_guard<std::mutex> lk(m_mtx);
    m_send_queue.push_back(std::move(buffer));
    if (m_on_writing)
        return;
    asyncWriteInternal();
}

void ClientChain::asyncRead()
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

void ClientChain::onReadSome()
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
        BOOST_LOG_TRIVIAL(error) << __LINE__ << ":ClientChain::onReadSome onReadSome " << head.m_seq_id;
        return;
    }

    auto cntl = it->second;
    cntl->appendReceiveData(m_receive_buffer.getReadPtr(), m_receive_buffer.getActiveSize());
    cntl->done();
}

void ClientChain::parseMessage()
{
    while (true) {
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

        if (m_receive_buffer.getActiveSize() < head.m_total_length) {
            break;
        }

        fslib::grpc::rpc_meta meta;
        if (!meta.ParseFromArray(ptr, head.m_meta_length)) {
            //TODO ½âÎöÊ§°Ü£¬closed
        }
        ptr += head.m_meta_length;

        auto request = std::make_unique<RpcRequest>(shared_from_this());
        {
            auto& msg = request->getRpcMessage();
            msg.setSeqId(meta.id());
            msg.setServiceName(meta.name());
            msg.setMessageBuffer(ptr, head.m_total_length - MESSAGE_HEAD_SIZE - head.m_meta_length);
        }
        m_receive_buffer.readCompleted(head.m_total_length);
    }
}

void ClientChain::asyncWriteInternal()
{
    if (m_send_buffer.getActiveSize() == 0 && m_send_queue.empty()) {
        m_on_writing = false;
        return;
    }
    if (m_send_buffer.getActiveSize() == 0 && !m_send_queue.empty()) {
        m_send_buffer = std::move(m_send_queue.front());
        m_send_queue.pop_front();
    }

    auto self = shared_from_this();
    m_socket.async_write_some(boost::asio::buffer(m_send_buffer.getReadPtr(), m_send_buffer.getActiveSize()),
        [this, self](boost::system::error_code ec, size_t length)
    {
        std::lock_guard<std::mutex> lk(m_mtx);
        if (!ec) {
            m_send_buffer.readCompleted(length);
            asyncWriteInternal();
        }
        else {
            //TODO
        }
    });
    m_on_writing = true;
}

}
}