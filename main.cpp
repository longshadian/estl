#include <iostream>
#include <vector>
#include <map>

#include "dynamic.hpp"

int main(int argc, char** argv)
{
    estl::Dynamic d;
    d = 12;
    d = 123;
    d = 1234.4133;
    std::cout << d.as_double() << std::endl;

    //std::map<estl::Dynamic, estl::Dynamic> m = {{"a", "123"}, {"b", "456"}};
    //std::cout << m.size() << std::endl;

	return 0;
}
