#include <iostream>
#include <string>
#include <algorithm>

size_t strReplace(std::string& str, char src, char dest)
{
    size_t t = 0;
    std::transform(str.begin(), str.end(), str.begin(),
        [&t, src, dest](char& c)
        {
            if (c == src) {
                c = dest;
                ++t;
            }
			return c;
        } );
    return t;
}


int main()
{
	std::string str = "a\nb\nc\nd\n";
	auto t = strReplace(str, 'd', ' ');

	std::cout << str.size() << " " << t << " " << str << "\n";

	return 0;
}

