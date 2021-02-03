#include "Strand.h"
#include "IOContext.h"

int main()
{
    try {
        // IOContext_Test1();
        // IOContext_Test2();
        Strand_Test();
    } catch (const std::exception& e) {
        std::cout << "exception: "<< e.what() << "\n";
        return -1;
    }
    std::cout << "main exit!\n";
    return 0;
}
