#pragma once

#include <unordered_map>
#include <numeric>
#include <list>
#include <boost/asio.hpp>

#include "RWHandler.h"

namespace zylib {

class Server
{
    static const int MAX_CONNECTION_NUM = 65536;
public:
    Server(boost::asio::io_service& io_service, short port);
    ~Server() = default;

    void accept();
private:
    void handleAcceptError(const boost::system::error_code& ec);
    void stopAccept();
    std::shared_ptr<RWHandler> createHandler();
    void reuseConnID(int conn_id);

    static void refuseAccept(boost::asio::ip::tcp::socket socket);
private:
    boost::asio::io_service& m_io_service;
    boost::asio::ip::tcp::acceptor m_acceptor;
    boost::asio::ip::tcp::socket m_socket;
    std::unordered_map<int, std::shared_ptr<RWHandler>> m_handlers;
    std::list<int> m_conn_id_pools;
};

}
