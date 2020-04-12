#pragma once

#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/log/trivial.hpp>

using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;

#define LOG BOOST_LOG_TRIVIAL

template <typename T>
struct X;
