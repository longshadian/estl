#pragma once

#include <cstdint>
#include <chrono>
#include <memory>
#include <vector>
#include <string>

#include "TcpHandler.h"

class TcpServerEvent
{
public:
    TcpServerEvent() = default;
    virtual ~TcpServerEvent() = default;
    TcpServerEvent(const TcpServerEvent& rhs) = delete;
    TcpServerEvent& operator=(const TcpServerEvent& rhs) = delete;
    TcpServerEvent(TcpServerEvent&& rhs) = delete;
    TcpServerEvent& operator=(TcpServerEvent&& rhs) = delete;

    // �µ�handler������
    virtual void OnAccept(TcpHdl hdl);

    // handler�ر�
    virtual void OnClosed(TcpHdl hdl);

    // handler��ʱ
    virtual void OnTimeout(TcpHdl hdl);

    // server���Ե�accept��handler��������
    virtual void OnAcceptOverflow();

    virtual void OnCatchException(const std::exception& e);

    virtual void OnDecode(TcpHdl hdl, ByteBuffer& buffer);
};

