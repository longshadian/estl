#include <cstdio>
#include <cassert>
#include <chrono>
#include <array>
#include <sstream>
#include <iomanip>
#include "Openssl.h"

std::string ToMD5String(const void* d, std::size_t len)
{
    const unsigned char* p = reinterpret_cast<const unsigned char*>(d);
    std::ostringstream sout;
    sout << std::hex << std::setfill('0');
    for (std::size_t i = 0; i != len; ++i) {
        unsigned char c = p[i];
        sout << std::setw(2) << (int)c;
    }
    return sout.str();
}

void Test()
{
    std::string path = R"(C:\Users\admin\Desktop\nginx-1.17.0\nginx-1.17.0\data\download\win7.iso)";
    auto* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        printf("ERROR: open file:[%s] failed.\n", path.c_str());
        return;
    }

    MD5state_st ctx{};
    std::memset(&ctx,0, sizeof(ctx));
    if (!::MD5_Init(&ctx)) {
        printf("ERROR: md5 init failed.\n");
        return;
    }

    std::array<char, 1024 * 16> buffer{};
    std::int64_t total = 0;
    while (true) {
        int32_t readn = std::fread(buffer.data(), 1, buffer.size(), f);
        if (readn < 0) {
            printf("ERROR: open file:[%s] failed.\n", path.c_str());
            break;
        }
        if (readn == 0) {
            printf("read file finished. [%s] size:[%lld]\n", path.c_str(), total);
            break;
        }
        total += readn;
        assert(::MD5_Update(&ctx, buffer.data(), readn));
    }

    std::array<unsigned char, MD5_DIGEST_LENGTH> md5_buffer{};
    assert(::MD5_Final(md5_buffer.data(), &ctx));

    auto md5_str = ToMD5String(md5_buffer.data(), md5_buffer.size());
    printf("md5: %s\n", md5_str.c_str());
    std::fclose(f);
}

void Test2()
{
    std::array<char, 10> buff{};
    auto s = ToMD5String(buff.data(), buff.size());
    printf("%d       %s\n", s.size(), s.c_str());
}

int main()
{
    Test2();
    system("pause");
    return 0;
}

