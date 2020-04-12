#include "Connector.h"

#include <iostream>

void ClientEvent::OnConnect(const boost::system::error_code& ec)
{
    if (ec) {
        std::cout << "ERROR: " << ec.message() << "\n";
    } else {
        std::cout << "connect success\n";
    }
}

void ClientEvent::OnRead(ClientMsgHead& head, std::vector<std::byte> body)
{
    auto msg = std::make_shared<Message>();
    std::memcpy(&msg->m_head, &head, sizeof(head));
    msg->m_body = std::move(body);

    auto it = m_handlerMap.find(msg->m_head.m_msg_id);
    if (it == m_handlerMap.end()) {
        std::cout << "can't find handler: " << msg->m_head.m_msg_id << "\n";
        return;
    }
    it->second(msg);
}

void ClientEvent::OnWrite()
{

}

void ClientEvent::OnClose(const boost::system::error_code* ec)
{
    if (ec == nullptr) {
        std::cout << "initiative close\n";
    } else {
        std::cout << "close: " << ec->message() << "\n";
    }
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

Connector::Connector(boost::asio::io_context& io_ctx)
    : m_io_ctx(io_ctx)
    , m_host()
    , m_port()
    , m_is_connected(false)
    , m_socket(m_io_ctx)
    , m_head()
    , m_body()
    , m_write_buffer()
    , m_event(std::make_unique<ClientEvent>())
{
}

bool Connector::Start(std::string host, uint16_t port)
{
    m_host = std::move(host);
    m_port = port;
    boost::asio::ip::tcp::endpoint addr(boost::asio::ip::address::from_string(m_host), port);
    m_socket.async_connect(addr, [this](const boost::system::error_code& ec) 
        {
            m_event->OnConnect(ec);
            if (ec) {
                m_is_connected = false;
                return;
            }
            m_is_connected = true;
            DoRead();
        });
    return m_is_connected;
}

bool Connector::isConnected() const
{
    return m_is_connected;
}

void Connector::sendMsg(const std::vector<char>& msg)
{
    m_io_ctx.post([this, msg]()
    {
        bool wait_write = !m_write_buffer.empty();
        m_write_buffer.push_back(std::move(msg));
        if (!wait_write) {
            DoWrite();
        }
    });
}

void Connector::sendMsg(const std::string& msg)
{
    std::vector<char> buff(msg.begin(), msg.end());
    sendMsg(buff);
}

void Connector::AysncClose()
{
    if (!m_is_connected.exchange(false))
        return;
    m_io_ctx.post([this]()
    {
        SyncClose(nullptr);
    });
}

void Connector::DoRead()
{
    m_socket.async_read_some(boost::asio::buffer(&m_head, sizeof(m_head)),
        [this](boost::system::error_code ec, size_t length) 
        {
            if (ec) {
                std::cout << "read ec:" << ec.value() << " " << ec.message() << std::endl;
                m_socket.close();
                return;
            }

            if (m_head.m_length > 1024 * 1024) {
                AysncClose();
                return;
            }
            int32_t body_size = m_head.m_length - sizeof(m_head);
            m_body.resize(body_size);
            DoReadBody();
        });
}

void Connector::DoReadBody()
{
    if (m_body.empty()) {
        m_event->OnRead(m_head, std::move(m_body));
        return DoRead();
    }

    m_socket.async_read_some(boost::asio::buffer(m_body),
        [this](boost::system::error_code ec, size_t length) 
        {
            if (ec) {
                SyncClose(&ec);
                return;
            }
            m_event->OnRead(m_head, std::move(m_body));
            DoRead();
        });
}

void Connector::DoWrite()
{
    boost::asio::async_write(m_socket,
        boost::asio::buffer(m_write_buffer.front().data(),
            m_write_buffer.front().size()),
        [this](boost::system::error_code ec, std::size_t /*length*/)
    {
        if (ec) {
            SyncClose(&ec);
            m_is_connected = false;
            return;
        }

        m_write_buffer.pop_front();
        if (!m_write_buffer.empty()) {
            DoWrite();
        }
    });
}

void Connector::SyncClose(const boost::system::error_code* p_ec)
{
    m_event->OnClose(p_ec);
    boost::system::error_code ec{};
    m_socket.shutdown(boost::asio::socket_base::shutdown_both, ec);
    m_socket.close(ec);
    m_is_connected = false;
}
