#include <string>
#include <iostream>
#include <chrono>
#include <thread>
#include <memory>

int TestH264_To_YUV();
void TestH264_Decoder();
void TestH264_DecoderTest2();

int TestMain()
{
    int n = 0;
    while (n < 5) {
        ++n;
        printf("Hello world! %d\n", n);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "main exit!!!!!!!!!\n";
    return 0;
}

int main()
{
    //TestH264_To_YUV();
    //TestH264_Decoder();
    TestH264_DecoderTest2();
    return 0;
}


