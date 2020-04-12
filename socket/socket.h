#pragma once

#ifdef _WIN32
#include <winsock2.h>
#else
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/tcp.h>

#define SOCKET int
#endif

#include <string>

using namespace std;

namespace sgs {

class Socket {
public:
    int timeout = 30;
    unsigned int max_send_buffer_size = 0xFFFF;
    unsigned int max_recv_buffer_size = 0xFFFF;

    SOCKET m_sock;

    Socket(int socketType = SOCK_STREAM);
    virtual ~Socket();

    virtual bool Connect(const string& host, unsigned short port);
    virtual bool IsConnected();
    virtual int  SocketType();
    virtual bool Bind(unsigned short nPort);
    virtual bool Accept(Socket& client);
    virtual void Close();

    virtual long Send(const char* buf, long buflen);
    virtual long Recv(char* buf, long buflen);
    virtual long SendTo(const char* buf, int len, const struct sockaddr_in* toaddr, int tolen);
    virtual long RecvFrom(char* buf, int len, struct sockaddr_in* fromaddr, int* fromlen);

    virtual bool GetPeerName(string& strIP, unsigned short &nPort);
    virtual void SetKeepAlive();

    SOCKET GetHandle();
private:
    int socketType;
    bool connected;

    string host;
    unsigned short port;
};

}
