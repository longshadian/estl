
#include <boost/static_string.hpp>

#include <iostream>
#include <string>

template <std::size_t N>
static void print_static_string(const boost::static_string<N>& s)
{
    std::cout << "c_str():          " << s.c_str() << "\n";
    std::cout << "capacity():       " << s.capacity() << "\n";
    std::cout << "empty():          " << s.empty() << "\n";
    std::cout << "szie():           " << s.size() << "\n";
    std::cout << "length():         " << s.length() << "\n";
    std::cout << "max_size():       " << s.max_size() << "\n";
    std::cout << "back():           " << s.back() << "\n";
    std::cout << "front():          " << s.front() << "\n";
}

static void fun()
{
    boost::static_string<7> s{};
    s.append("12345");
    print_static_string(s);
    std::cout << s.c_str() << "\n";
    s.assign("1234567");
    print_static_string(s);
    std::cout << s.c_str() << "\n";
}


int main()
{
    fun();
    return 0;
}
