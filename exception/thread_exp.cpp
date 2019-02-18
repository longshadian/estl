#include <iostream>
#include <chrono>
#include <thread>
#include <exception>
#include <cassert>

void fun()
{
    try {
        int n = 0;
        while (n < 3) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            n++;
        }
        throw std::runtime_error("hahah");
       } catch(const std::exception& e) {
           assert(0);
       }
}

int main()
{
    std::thread d{&fun}; 
    d.join();
    return 0;
}
