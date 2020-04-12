#include <errno.h>
#include <cassert>

#include "socket.h"

namespace sgs {

#ifndef _WIN32
#define strnicmp strncasecmp
#endif

Socket::Socket(int socketType) {
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
    if (LOBYTE(wsa_data.wVersion) != 2 || HIBYTE(wsa_data.wVersion) != 2) {
        WSACleanup();
    }
#endif

    this->socketType = socketType;
    port = 0;
    connected = false;

    if (socketType == SOCK_STREAM)
        m_sock = socket(AF_INET, SOCK_STREAM, 0);
    else {
        m_sock = socket(AF_INET, SOCK_DGRAM, 0);

        if (m_sock != -1) {
            connected = true;
        }
    }

    assert(m_sock != -1);
}

Socket::~Socket() {
    Close();
#ifdef _WIN32
    WSACleanup();
#endif
}

SOCKET Socket::GetHandle() {
    return m_sock;
}

void Socket::Close() {
    if (m_sock != -1) {
#ifdef _WIN32
        shutdown(m_sock, SD_BOTH);
        closesocket(m_sock);
#else
        shutdown(m_sock, SHUT_RDWR);
        close(m_sock);
#endif
        m_sock = -1;
    }

    connected = false;
}

bool Socket::Connect(const string& host, unsigned short port) {
    if (m_sock == -1)
        return false;

    this->host = host;
    this->port = port;

    struct hostent * he = gethostbyname(host.c_str());

    if (he == NULL) {
        Close();
        return false;
    }

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_addr = *((struct in_addr*) he->h_addr);
    memset(sin.sin_zero, 0, 8);
    sin.sin_port = htons(port);

    setsockopt(m_sock, SOL_SOCKET, SO_SNDBUF, (char*)&max_send_buffer_size, sizeof(max_send_buffer_size));
    setsockopt(m_sock, SOL_SOCKET, SO_RCVBUF, (char*)&max_recv_buffer_size, sizeof(max_recv_buffer_size));

    if (connect(m_sock, (struct sockaddr *)&sin, sizeof(sin))) {
        Close();
        return false;
    }

    connected = true;
    return true;
}

long Socket::Send(const char* buf, long buflen) {
    if (m_sock == -1) {
        return -1;
    }

    long sended = 0;

    do {
        long len = send(m_sock, buf + sended, buflen - sended, 0);
        if (len < 0) {
            break;
        }
        sended += len;
    } while (sended < buflen);

    return sended;
}

long Socket::Recv(char* buf, long buflen) {
    if (m_sock == -1) {
        return -1;
    }

    fd_set fd;
    FD_ZERO(&fd);
    FD_SET(m_sock, &fd);
    struct timeval val = { timeout, 0 };
    int selret = select(m_sock + 1, &fd, NULL, NULL, &val);

    if (selret <= 0) {
        return selret;
    }

    long len = recv(m_sock, buf, buflen, 0);

    return len;
}

bool Socket::GetPeerName(string& strIP, unsigned short &nPort) {
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    int addrlen = sizeof(addr);
#ifdef WIN32
    if(getpeername(m_sock, (struct sockaddr*)&addr, &addrlen)!=0)
#else
    if (getpeername(m_sock, (struct sockaddr*) &addr, (socklen_t*) &addrlen) != 0)
#endif
        return false;

    char szIP[64];
#ifdef WIN32
    sprintf(szIP, "%u.%u.%u.%u", addr.sin_addr.S_un.S_addr & 0xFF, (addr.sin_addr.S_un.S_addr >> 8) & 0xFF, (addr.sin_addr.S_un.S_addr >> 16) & 0xFF, (addr.sin_addr.S_un.S_addr >> 24) & 0xFF);
#else
    sprintf(szIP, "%u.%u.%u.%u", addr.sin_addr.s_addr & 0xFF, (addr.sin_addr.s_addr >> 8) & 0xFF, (addr.sin_addr.s_addr >> 16) & 0xFF, (addr.sin_addr.s_addr >> 24) & 0xFF);
#endif
    strIP = szIP;
    nPort = ntohs(addr.sin_port);

    return true;
}

bool Socket::IsConnected() {
    return (m_sock != -1) && connected;
}

int Socket::SocketType() {
    return socketType;
}

long Socket::SendTo(const char* buf, int len, const struct sockaddr_in* toaddr, int tolen) {
    if (m_sock == -1) {
        return -1;
    }

    return sendto(m_sock, buf, len, 0, (const struct sockaddr*)toaddr, tolen);
}

long Socket::RecvFrom(char* buf, int len, struct sockaddr_in* fromaddr, int* fromlen) {
    if (m_sock == -1) {
        return -1;
    }

#ifdef WIN32
    return recvfrom(m_sock,buf,len,0,(struct sockaddr*)fromaddr,fromlen);
#else
    return recvfrom(m_sock, buf, len, 0, (struct sockaddr*)fromaddr, (socklen_t*)fromlen);
#endif
}

bool Socket::Bind(unsigned short nPort) {
    if (m_sock == -1) {
        return false;
    }

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
#ifdef WIN32
    sin.sin_addr.S_un.S_addr = 0;
#else
    sin.sin_addr.s_addr = 0;
#endif
    memset(sin.sin_zero, 0, 8);
    sin.sin_port = htons(nPort);

    if (bind(m_sock, (sockaddr*)&sin, sizeof(sockaddr_in)) != 0)
        return false;
    listen(m_sock, 1024);

    connected = true;
    return true;
}

bool Socket::Accept(Socket& client) {
    if (m_sock == -1) {
        return false;
    }

    client.m_sock = accept(m_sock, NULL, NULL);
    client.connected = true;

    return (client.m_sock != -1);
}

void Socket::SetKeepAlive() {
    int alive = 1;
    setsockopt(m_sock, SOL_SOCKET, SO_KEEPALIVE, (const char *)&alive, sizeof(alive));

//    int idle = 10, intv = 5, cnt = 3;
//    setsockopt(m_sock, SOL_TCP, TCP_KEEPIDLE,  &idle, sizeof(idle));
//    setsockopt(m_sock, SOL_TCP, TCP_KEEPINTVL, &intv, sizeof(intv));
//    setsockopt(m_sock, SOL_TCP, TCP_KEEPCNT,   &cnt,  sizeof(cnt));
}

}
