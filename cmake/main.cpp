#include <vector>
#include <iostream>

#include "x.h"
#include "a.h"
//#include "c.h"

int main()
{
    std::vector<int> v = {1, 2};
    std::cout << v.size() << "\n";
    fun(33);
    a();
    //c();
    return 0;
}
