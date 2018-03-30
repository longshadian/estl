#include <iostream>
#include <string>

#include <json/json.h>

using namespace Json;

int main()
{
    //std::string s = R"({"a":1,"b":{"c":2}})";
    std::string s = R"({"a":1,"b":2.3})";
    Value v;
    Reader reader;
    if (!reader.parse(s, v)) {
        std::cout << "parse error\n";
        return 0;
    }
    std::cout << s << "\n";

    try {
        std::cout << "isMember a:" << v.isMember("a") << "\n";
        std::cout << "isMember b:" << v.isMember("b") << "\n";
        std::cout << "isMember d:" << v.isMember("d") << "\n";
        std::cout << v["a"].isNumeric() << "\n";
        std::cout << v["b"].isNumeric() << "\n";
        if (v["b"].isObject())
            std::cout << "isMember c:" << v["b"].isMember("c") << "\n";
    } catch (const std::exception& e) {
        std::cout << "excpetion:" << e.what() << "\n";
    }
    return 0;
}