#include "tests/TestH264File.h"
#include "tests/TestHEVCFile.h"
#include "tests/TestRtsp_Live555.h"
#include "tests/TestRtsp_FFMpeg.h"

int main()
{
    //TestH264File();
    //TestHEVCFile();
    //TestRtsp_Live555();
    TestRtsp_FFMpeg();

    return 0;
}