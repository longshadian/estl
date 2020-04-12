#pragma once

#include <boost/beast/version.hpp>
#include <cstdlib>
#include <iostream>
#include <string>

#include "Types.h"

class HttpClient
{
public:
    HttpClient();
    ~HttpClient();
    HttpClient(const HttpClient&) = delete;
    HttpClient& operator=(const HttpClient&) = delete;
    HttpClient(HttpClient&&) = delete;
    HttpClient& operator=(HttpClient&&) = delete;

    http::response<http::string_body> Request(http::verb method, const char* host, unsigned short port, std::string_view path
        , http::request<http::empty_body> req);

    void Reset();

private:
    boost::asio::io_context         m_ioc;
    boost::asio::ip::tcp::resolver  m_resolver;
    boost::asio::ip::tcp::socket    m_socket;
};
