#include <iostream>

#include "Executor.h"

int main()
{
    try {
        Executor_Test();
    } catch (const std::exception& e) {
        std::cout << "exception: "<< e.what() << "\n";
        return -1;
    }
    std::cout << "main exit!\n";
    return 0;
}
