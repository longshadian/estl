#include "xdecode/XDecode.h"

#include <string>
#include <cstdio>
#include <array>
#include "console_log.h"


static void Test_LocalFile()
{
    std::string local_file = "D:/resource/xiaoen.h264";
    FILE* f = ::fopen(local_file.c_str(), "rb");
    if (!f) {
        CONSOLE_LOG_WARN << "open file failure.  " << local_file;
        return;
    }

    std::array<char, 1024 * 4> buf;
    while (1) {
        auto readn = fread(buf.data(), 1, buf.size(), f);
        if (readn == 0)
            break;

        unsigned char nal = buf[4];
        int nal_unit_type = nal & 0x1F;
        int nal_ref_bit = (nal >> 5) & 0x02;
        int nal_forbidden_bit = (nal >> 7);
        logPrintInfo("head: %x %x %x %x %x %x", buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]);
        logPrintInfo("head: %d %d %d", nal_forbidden_bit, nal_ref_bit, nal_unit_type);
        if (1)
            break;
    }
    fclose(f);
}

void Test_XDecode_Load264()
{
    Test_LocalFile();
}

