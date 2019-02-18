#include <iostream>       // std::cout
#include <functional>     // std::ref
#include <thread>         // std::thread
#include <future>
#include <chrono>
#include <exception>

int main()
{
    std::future<int> f{};
    try {
        auto p = std::make_shared<std::promise<int>>();
        f = p->get_future();
        //p->set_value(2222);
        p->set_exception(std::current_exception());
        p = nullptr;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    } catch (const std::exception& e) {
        std::cout << e.what() << " xxxx\n";
        return 0;
    }

    try {
        std::cout << f.get() << "\n";
    } catch (const std::exception& e) {
        std::cout << e.what() << "\n";
    }
    return 0;
}