//
// blocking_tcp_echo_client.cpp
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
// Copyright (c) 2003-2015 Christopher M. Kohlhoff (chris at kohlhoff dot com)
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <chrono>
#include <boost/asio.hpp>

using boost::asio::ip::tcp;

enum { max_length = 1024 };

int main(int argc, char* argv[])
{
  try
  {
    boost::asio::io_service io_service;

    tcp::socket s(io_service);
    tcp::resolver resolver(io_service);
    boost::asio::connect(s, resolver.resolve({"127.0.0.1", "21010"}));

    while (true) {
        std::cout << "message:  ";
        std::array<char, max_length> request{};
        request.fill(0);
        std::cin.getline(request.data(), max_length);

        int32_t request_length = (int32_t)std::strlen(request.data());
        if (request_length > 0) {
            auto send_tm = std::chrono::system_clock::now();
            std::array<int32_t, 2> head_buffer{0};
            head_buffer.fill(0);
            head_buffer[0] = request_length + 8;
            head_buffer[1] = 1;
            boost::asio::write(s, boost::asio::buffer(head_buffer.data(), 8));
            boost::asio::write(s, boost::asio::buffer(request.data(), request_length));

            head_buffer.fill(0);
            char reply_head[8] = {0};
            boost::asio::read(s, boost::asio::buffer(reply_head, 8));
            std::memcpy(&head_buffer[0], reply_head, 4);
            std::memcpy(&head_buffer[1], reply_head + 4, 4);

            if (4 < head_buffer[0] && head_buffer[0] < 100) {

                std::vector<char> reply_body;
                reply_body.resize(head_buffer[0] - 8);
                size_t reply_length = boost::asio::read(s,
                    boost::asio::buffer(reply_body));
                std::cout << "received: ";
                std::cout.write(reply_body.data(), reply_body.size());

                std::cout << "cost:" << 
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now() - send_tm).count();
                std::cout << "\n\n";
            }
        }
    }
  } catch (std::exception& e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}
