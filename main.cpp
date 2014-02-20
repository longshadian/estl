#include <iostream>
#include <vector>

#include "dynamic.hpp"

int main(int argc, char** argv)
{
    estl::Dynamic d;

    std::vector<int> v = {1, 2, 3, 4};
    for (auto i : v)
        std::cout << i << std::endl;

	return 0;
}
