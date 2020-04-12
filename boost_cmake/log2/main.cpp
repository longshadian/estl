#include <string>
#include <iostream>

#include <boost/log/trivial.hpp>

#include <boost/asio.hpp>

int main() 
{
    BOOST_LOG_TRIVIAL(trace) << "A trace severity message";
    BOOST_LOG_TRIVIAL(debug) << "A debug severity message";
    BOOST_LOG_TRIVIAL(info) << "An informational severity message";
    BOOST_LOG_TRIVIAL(warning) << "A warning severity message";
    BOOST_LOG_TRIVIAL(error) << "An error severity message";
    BOOST_LOG_TRIVIAL(fatal) << "A fatal severity message";

    BOOST_LOG_TRIVIAL(debug) << "start test";

    boost::asio::io_context io_context{};
    auto work_guard = boost::asio::make_work_guard(io_context);
    io_context.run();

    BOOST_LOG_TRIVIAL(debug) << "finish test";

    system("pause");

    return 0;
}