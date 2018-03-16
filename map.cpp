#include <map>
#include <iostream>

int main()
{

    std::map<int, int> v{};
	v[0] = 0;
	v[1] = 1 * 2;
	v[2] = 2 * 2;

std::cout << v.erase(0) << "\n";
std::cout << v.erase(0) << "\n";
    return 0;
}
