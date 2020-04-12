#pragma once

#include <string>
#include <array>
#include <atomic>
#include <list>
#include <boost/asio.hpp>

class RWHandler;

#pragma pack(push, 1)
struct ClientMsgHead
{
    int32_t             m_length;
    uint8_t             m_sequence_id;
    uint8_t             m_unknown_1;
    uint8_t             m_unknown_2;
    uint8_t             m_unknown_3;
    std::array<char, 16> m_sid;
    int32_t             m_msg_id;
};
#pragma pack(pop)

struct Message
{
    ClientMsgHead           m_head;
    std::vector<std::byte>  m_body;
};

class ClientEvent
{
public:
    ClientEvent() = default;
    virtual ~ClientEvent() = default;
    ClientEvent(const ClientEvent&) = delete;
    ClientEvent& operator=(const ClientEvent&) = delete;
    ClientEvent(ClientEvent&&) = delete;
    ClientEvent& operator=(ClientEvent&&) = delete;

    virtual void OnConnect(const boost::system::error_code& ec);
    virtual void OnRead(ClientMsgHead& head, std::vector<std::byte> v);
    virtual void OnWrite();
    virtual void OnClose(const boost::system::error_code* ec);

private:
    std::unordered_map<int32_t, std::function<void(std::shared_ptr<Message>)>> m_handlerMap;
};


class Connector
{
    static const int READ_BUFFER_LENGTH = 1024;
public:
    Connector(boost::asio::io_context& ios);
    ~Connector() = default;

    bool Start(std::string host, uint16_t port);
    bool isConnected() const;
    void sendMsg(const std::vector<char>& msg);
    void sendMsg(const std::string& msg);
    void AysncClose();
private:
    void DoRead();
    void DoReadBody();
    void DoWrite();
    void SyncClose(const boost::system::error_code* p_ec);

private:
    boost::asio::io_context&        m_io_ctx;
    std::string                     m_host;
    uint16_t                        m_port;
    std::atomic<bool>               m_is_connected;
    boost::asio::ip::tcp::socket    m_socket;

    ClientMsgHead                   m_head;
    std::vector<std::byte>          m_body;

    std::list<std::vector<char>>    m_write_buffer;
    std::shared_ptr<ClientEvent>    m_event;
};
