

#include <ios>
#include <iostream>
#include <string>

#include <boost/format.hpp>

void BoostFormat1(boost::format& fmt)
{
}

template <typename T, typename... Args>
void BoostFormat1(boost::format& fmt, T&& v, Args&&... args)
{
	fmt % v;
	BoostFormat1(fmt, args...);
}

template <typename... Args>
std::string BoostFormat(const char* fmt, Args... args)
{
	try {
		boost::format formater{ fmt };
		BoostFormat1(formater, std::forward<Args&&>(args)...);
		return formater.str();
	} catch (...) {
		return {};
	}
}

int main()
{
    try {
        auto f = boost::format("xxx_%1%_%2%.log") % "11234" % "%Y_%m.log";
        std::cout << f.str() << "\n";

		std::cout << BoostFormat("aaaa %1%", 123) << "\n";
		std::cout << BoostFormat("bbb") << "\n";

    } catch (std::exception& e) {
        std::cout << "exception: " << e.what() << "\n";
    }

    return 0;
}
