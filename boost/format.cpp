#include <string>
#include <iostream>

#include <boost/format.hpp>

int main()
{
	std::cout << boost::format("%1%%%\n") % 222;
	return 0;
}
