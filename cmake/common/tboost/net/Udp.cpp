#include "net/Udp.h"

#include <iostream>

UdpClient::UdpClient(std::string host, uint16_t port)
    : m_io()
    , m_socket(m_io)
    , m_remote_endpoint()
    , m_read_length()
{
    m_socket.open(boost::asio::ip::udp::v4());
    boost::asio::ip::udp::resolver resolver{m_io};
    m_remote_endpoint = *resolver.resolve({ host, std::to_string(port) });
}

UdpClient::UdpClient(std::uint16_t local_port, std::string host, std::uint16_t port)
    : m_io()
    , m_socket(m_io, boost::asio::ip::udp::endpoint{boost::asio::ip::udp::v4(), local_port})
    , m_remote_endpoint()
    , m_read_length()
{
    boost::asio::ip::udp::resolver resolver{m_io};
    m_remote_endpoint = *resolver.resolve({ host, std::to_string(port) });
}

UdpClient::~UdpClient()
{

}

std::int32_t UdpClient::GetPacketBlocking(void* data, std::size_t len, std::int32_t millisec)
{
    m_io.restart();
    m_read_length = 0;
    m_socket.async_receive_from(boost::asio::buffer(data, len), m_remote_endpoint
        , std::bind(&UdpClient::OnRead, this, std::placeholders::_1, std::placeholders::_2));
    m_io.run_one_for(std::chrono::milliseconds{ millisec });
    return m_read_length;
}

std::int32_t UdpClient::SendPacket(const void* data, std::size_t len)
{
    return static_cast<std::int32_t>(m_socket.send_to(boost::asio::buffer(data, len), m_remote_endpoint));
}

void UdpClient::OnRead(boost::system::error_code ec, std::size_t length)
{
    if (ec) {
        std::cout << "client onread error " << ec.message() << "\n";
        return;
    }
    if (ec) {
        m_read_length = -1;
    }
    std::cout << "client onread " << ec.message() << "\n";
    m_read_length = static_cast<std::int32_t>(length);
}


//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

UdpServer::UdpServer(boost::asio::io_context& io_context, std::uint16_t port)
    : m_socket(io_context, boost::asio::ip::udp::endpoint(boost::asio::ip::udp::v4(), port))
{
    StartReceive();
}

UdpServer::~UdpServer()
{
}

void UdpServer::StartReceive()
{
    m_socket.async_receive_from(
        boost::asio::buffer(m_recv_buffer), m_remote_endpoint,
        std::bind(&UdpServer::OnRead, this,
            std::placeholders::_1,
            std::placeholders::_2));
}

void UdpServer::OnRead(const boost::system::error_code& error,
    std::size_t bytes_transferred)
{
    if (!error) {
        /*
        auto msg = std::make_shared<std::string>("from server");
        m_socket.async_send_to(boost::asio::buffer(*msg), m_remote_endpoint,
            std::bind(&UdpServer::OnWrite, this, msg,
                std::placeholders::_1,
                std::placeholders::_2));
                */
        std::string s{m_recv_buffer.begin(), m_recv_buffer.begin() + bytes_transferred};
        auto ip = m_remote_endpoint.address().to_string();
        auto port = m_remote_endpoint.port();
        printf(">>>>>>>>>>>>>>server OnRead %s  len: %d client socket: %s : %d\n"
            , s.c_str(), (int)bytes_transferred, ip.c_str(), port);
        StartReceive();
    }
}

void UdpServer::OnWrite(std::shared_ptr<std::string> msg,
    const boost::system::error_code& /*error*/,
    std::size_t /*bytes_transferred*/)
{

}
