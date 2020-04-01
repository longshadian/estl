#include "Snowflake.h"

#include <string>
#include <iostream>

void TestSnowflake()
{
    Snowflake<41,10,12> sf{10};
    std::cout << "MAX_TIMESTAMP:" << sf.MAX_TIMESTAMP() << "\n";
    std::cout << "MAX_WORKER_ID:" << sf.MAX_WORKER_ID() << "\n";
    std::cout << "MAX_SEQUENCE:" << sf.MAX_SEQUENCE() << "\n";

    /*
    int cnt = 0;
    int64_t last_value = 0;
    while (true) {
        int64_t value = sf.NewID();
        if (value == 0)
            break;
        last_value = value;
        std::cout << last_value << "\n";
        ++cnt;
    }
    std::cout << "count: " << cnt << "\n";
    auto arr = sf.Parse(last_value);
    std::cout << last_value << "\t" << arr[0] << "\t" << arr[1] << "\t" << arr[2] << "\n";
    */

    /*
    for (int i = 0; i != 10; ++i) {
        int64_t value = sf.NewID();
        int64_t value = sf.NewID();
        auto arr = sf.Parse(value);
        std::cout << value << "\t" << arr[0] << "\t" << arr[1] << "\t" << arr[2] << "\n";
    }
    */
    auto tbegin = std::chrono::steady_clock::now();
    int cnt = 100000;
    for (int i = 0; i != cnt; i++) {
        auto value = sf.NewID();
        if (value == 0) {
            std::cout << "break: \n"; 
            break;
        }
    }
    auto tend = std::chrono::steady_clock::now();

    auto cost = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();

    std::cout << "cost: " << (cost / (float)cnt) << "\n";
}

int main()
{
    TestSnowflake();

    system("pause");
    return 0;
}
