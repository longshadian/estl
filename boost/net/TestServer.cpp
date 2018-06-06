#include <iostream>

#include "Message.h"
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

    }

    virtual void OnClosed(TcpHdl hdl) override
    {

    }

    virtual void OnTimeout(TcpHdl hdl) override
    {

    }

    virtual void OnAcceptOverflow() override
    {

    }

    virtual void OnCatchException(const std::exception& e) override
    {

    }

    virtual void OnDecode(TcpHandler hdl, ByteBuffer& buffer) override
    {

    }
};

int main() 
{
    return 0;
}
