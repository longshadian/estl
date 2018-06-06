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

    // 新的handler创建了
    virtual void OnAccept(TcpHdl hdl) = 0;

    // handler关闭
    virtual void OnClosed(TcpHdl hdl) = 0;

    // handler超时
    virtual void OnTimeout(TcpHdl hdl) = 0;

    // server可以得accept的handler超出上限
    virtual void OnAcceptOverflow() = 0;

    virtual void OnCatchException(const std::exception& e) = 0;

    virtual void OnDecode(TcpHdl hdl, ByteBuffer& buffer) = 0;
};

