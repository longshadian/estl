#include <memory>

int main()
{

	auto v = std::make_shared<int>(1233);
	++*v;
	++*v;
	++*v;
	++*v;
	++*v;


return 0;
}

