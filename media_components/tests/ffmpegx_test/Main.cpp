
//void TestFileToMp4();
int TestMP4(int argc, char **argv);
int TestTranscodeing(int argc, char **argv);
int TestTranscode2();
int TestTranscodeXX();
int TestDecode();
int TestH264File();
int TestXFFmpeg_MemoryDemuxer();
int TestRtspPull();

int main(int argc, char **argv)
{
    //TestFileToMp4();
    //TestMP4(1, nullptr);
    //TestTranscodeing(1, nullptr);
    //TestTranscode2();
    //TestTranscodeXX();
    //TestDecode();
    //TestH264File();
    //TestXFFmpeg_MemoryDemuxer();

    TestRtspPull();

    return 0;
}


