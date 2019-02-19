
#include <iostream>
#include <memory>
#include <string>
#include <map>
#include <unordered_map>

int main()
{
    auto s1 = std::make_shared<std::string>("s1");
    //auto s2 = std::make_shared<std::string>("s2");

    auto w1_1 = std::weak_ptr<std::string>(s1);
    auto w1_2 = std::weak_ptr<std::string>(s1);

    std::cout << s1.unique() << "\n";
    std::cout << w1_1.expired() << "\n";
    std::cout << *w1_1.lock() << "\n";

    std::cout << "----free\n";
    /*
    std::cout << w1_1.expired() << "\n";
    std::cout << w1_1.lock() << "\n";
    */
    s1 = nullptr;

    std::map<std::weak_ptr<std::string>, std::string,
        std::owner_less<std::weak_ptr<std::string>>> m{};
    m[w1_1] = "w1_1";
    m[w1_2] = "w1_2";

    auto it1 = m.find(w1_1);
    auto it2 = m.find(w1_2);
    std::cout << "it equal " << (it1 == it2) << "\n";
    std::cout << "size:" << m.size() << "\n";

    return 0;
}
