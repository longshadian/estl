#pragma once

#include <cstdint>
#include <chrono>
#include <functional>
#include <string>
#include <array>
#include <vector>

#include <boost/asio.hpp>

class UdpClient
{
public:
    UdpClient(std::string host, std::uint16_t port);
    UdpClient(std::uint16_t local_port, std::string host, std::uint16_t port);
    ~UdpClient();
    UdpClient(const UdpClient&) = delete;
    UdpClient& operator=(const UdpClient&) = delete;

    std::int32_t GetPacketBlocking(void* data, std::size_t len, std::int32_t millisec);
    std::int32_t SendPacket(const void* data, std::size_t len);

    boost::asio::io_service         m_io;
    boost::asio::ip::udp::socket    m_socket;
    boost::asio::ip::udp::endpoint  m_remote_endpoint;
    std::int32_t                    m_read_length;

private:
    void OnRead(boost::system::error_code ec, std::size_t length);
};

class UdpServer
{
public:
    UdpServer(boost::asio::io_context& io_context, std::uint16_t port);
    ~UdpServer();
    UdpServer(const UdpServer&) = delete;
    UdpServer& operator=(const UdpServer&) = delete;

private:
    void StartReceive();
    void OnRead(const boost::system::error_code& error, std::size_t);
    void OnWrite(std::shared_ptr<std::string>, const boost::system::error_code&, std::size_t);

private:

    boost::asio::ip::udp::socket        m_socket;
    boost::asio::ip::udp::endpoint      m_remote_endpoint;
    std::array<char, 32>                m_recv_buffer;
};

