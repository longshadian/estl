#include <iostream>
#include <future>
#include <memory>
#include <functional>

using CB = std::function<void()>;

struct CallbackChain
{
    std::vector<CB> m_chain;
};

int main()
{

    return 0;
}