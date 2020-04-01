#include "zylib/zylib.h"

#include <iostream>

int main()
{
    using UUID_String = zylib::FixedString<12>;
    using UUID_String_Ex = zylib::FixedString<13>;

    UUID_String s1;
    zylib::bzero(&s1);

    UUID_String s2;
    zylib::bzero(&s2);

    UUID_String_Ex s100;
    zylib::bzero(&s100);

    std::cout << (s1 == s2) << "\n";
    std::cout << (s1 != s2) << "\n";
    //std::cout << (s1 != s100) << "\n";

    s1.setString("sssssa");

    std::cout << s1 << "\n";
    std::cout << s100 << "\n";

    return 0;
}
