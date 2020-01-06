#include <iostream>
#include <vector>
#include <map>

#include "dynamic.h"

int main(int argc, char** argv)
{
    estl::Dynamic d;
    d = 12;
    std::cout << d.asInt() << std::endl;
    d = estl::Dynamic(estl::Dynamic::Type::OBJECT);
    d["a"] = "adf";
    d["b"] = 333;
    estl::Dynamic& dd = d["c"];
    dd = estl::Dynamic(estl::Dynamic::ARRAY);
    dd.push_back(3);
    dd.push_back("sx");
    std::cout << d["a"].asString() << " "
        << d["b"].asInt()
        << std::endl;
    const estl::Dynamic& da = d["c"];
    std::cout << da.at(0).asInt() << " "
        << da.at(1).asString()
        << std::endl;
    estl::Dynamic darray(estl::Dynamic::ARRAY);
    darray.push_back(12);
    darray.push_back(d);
    estl::Dynamic& d3 = darray.at(1);
    std::cout << d3["a"].asString() << " " << d3["b"].asInt() << std::endl;
	return 0;
}
