#include <iostream>
#include <algorithm>
#include <list>
#include <vector>

struct T
{
	int a;
};

int main()
{
	std::list<T> lt = {{1}, {2}, {3}, {4}, {5}};

	T x = {3};
	auto it = 
	std::lower_bound(lt.begin(), lt.end(), x, [](const T& t, const T& val) { return t.a < val.a; });
	std::cout << it->a << "\n";

	lt.insert(lt.end(), {111});
	for (auto& l : lt)
		std::cout << l.a << " ";
	std::cout << "\n";



	return 0;
}

