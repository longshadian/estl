#include <iostream>

#include "TcpEvent.h"
#include "TcpServer.h"
#include "TcpHandler.h"
#include "ByteBuffer.h"

class ServerEvent : public TcpServerEvent
{
public:
    ServerEvent() = default;
    virtual ~ServerEvent() override = default;
    ServerEvent(const ServerEvent&) = delete;
    ServerEvent& operator=(const ServerEvent&) = delete;
    ServerEvent(ServerEvent&&) = delete;
    ServerEvent& operator=(ServerEvent&&) = delete;

    virtual void OnAccept(TcpHdl hdl) override
    {
        std::cout << "OnAccept:\n";
        auto handler = hdl.lock();
        std::cout << "OnAccept:" << handler->GetConnID() << "\n";
    }

    virtual void OnClosed(TcpHdl hdl) override
    {
        auto handler = hdl.lock();
        std::cout << "OnClosed:" << handler->GetConnID() << "\n";
    }

    virtual void OnTimeout(TcpHdl hdl) override
    {
        auto handler = hdl.lock();
        std::cout << "OnTimeout:" << handler->GetConnID() << "\n";
    }

    virtual void OnAcceptOverflow() override
    {
        std::cout << "OnAcceptOverflow:" << "\n";
    }

    virtual void OnCatchException(const std::exception& e) override
    {
        std::cout << "OnCatchException:" << e.what() << "\n";
    }

    virtual void OnDecode(TcpHdl hdl, ByteBuffer& buffer) override
    {
        size_t total = buffer.ReadableSize();
        std::cout << "OnDecode: " << total << "\n";
        for (size_t i = 0; i != total; ++i) {
            const uint8_t* pos = reinterpret_cast<const uint8_t*>(buffer.GetReaderPtr(i));
            if (*pos == '\n') {
                size_t src_len = i + 1;
                std::vector<uint8_t> src{};
                src.resize(src_len);
                buffer.ReadeArray(src.data(), src.size());

                auto handler = hdl.lock();
                handler->Send(std::make_shared<ByteBuffer>(std::move(src)));
            }
        }
    }
};

int main() 
{
    auto server = std::make_unique<TcpServer>(9900, std::make_unique<ServerEvent>(), TcpServer::Option{});
    server->Start();
    server->WaitThreadExit();
    return 0;
}
