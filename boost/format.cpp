

#include <ios>
#include <iostream>
#include <string>

#include <boost/format.hpp>

int main()
{
    try {
        auto f = boost::format("xxx_%1%_%2%.log") % "11234" % "%Y_%m.log";
        std::cout << f.str() << "\n";
    } catch (std::exception& e) {
        std::cout << "exception: " << e.what() << "\n";
    }

    return 0;
}
