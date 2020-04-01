#include <cstdio>
#include <cstdint>

//#define USE_TEST

#if defined (USE_TEST)
#include <endian.h>

TestCase("test Endian")
{
    uint64_t h = 0x1122334455667788;
    uint64_t e1 = htole64(h);
    uint64_t e2 = le64toh(e1);
    
    printf("%0lx, %0lx, %0lx\n", h, e1, e2);
    
    return 0;
}

#endif

