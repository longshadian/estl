
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <array>
    
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <zlib.h>

const std::string FILE_NAME = "./replay_server.x";
const size_t HEAD_SIZE = 8;

int32_t g_seq = 0;

void compressEx(void* p, size_t len);
void compressEx2(uint8_t* p, size_t len);
void gzCompress(const void* p, size_t len);

bool checkHead(const uint8_t* pos, size_t size)
{
    while (true) {
        if (size == 0) {
            std::cout << "size == 0\n";
            return true;
        }

        if (size < HEAD_SIZE) {
            std::cout << "size < HEAD_SIZE\n";
            return false;
        }
        if (size == HEAD_SIZE) {
            std::cout << "size == HEAD_SIZE\n";
            return true;
        }

        size_t len{};
        std::memcpy(&len, pos, 4);
        if (len < HEAD_SIZE) {
            std::cout << "len < HEAD_SIZE\n";
            return true;
        }
        if (size < len) {
            std::cout << "size < len\n";
            return false;
        }

        pos += 4;
        int32_t seq{};
        std::memcpy(&seq, pos,4);
        if (seq - g_seq != 1) {
            std::cout << "seq - g_seq != 1\n";
            return false;
        }

    }
}

void fun()
{
    boost::interprocess::file_mapping src_file{FILE_NAME.c_str(), boost::interprocess::read_only};
    boost::interprocess::mapped_region region{src_file, boost::interprocess::read_only};
    void* addr = region.get_address();
    size_t size = region.get_size();

    std::cout << "file size: " << size << "\n";
    gzCompress(addr, size);
}

void compressEx(void* p, size_t len)
{
    auto out_len = ::compressBound(len);
    std::vector<uint8_t> buffer{};
    buffer.resize(out_len);

    auto ret = ::compress(buffer.data(), &out_len, (Bytef*)p, len);
    if (ret != Z_OK) {
        std::cout << "compressEx " << ret << "\n";
        return;
    }

    std::string out_name = "./xx.zip";
    std::ofstream ofstream{};
    ofstream.open(out_name.c_str(), std::ios_base::binary);
    ofstream.write((const char*)buffer.data(), out_len);
}

void compressEx2(uint8_t* p, size_t len)
{
    z_stream strm;
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    auto ret = ::deflateInit(&strm, Z_DEFAULT_COMPRESSION);

    std::array<uint8_t, 512> temp{};

    std::string out_name = "./xx.zip";
    std::ofstream ofstream{};
    ofstream.open(out_name.c_str(), std::ios_base::binary);

    strm.avail_in = len;
    strm.next_in = p;
    do {
        strm.avail_out = temp.size();
        strm.next_out = temp.data();

        ret = ::deflate(&strm, Z_FINISH);    /* no bad return value */
        assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
        auto have = 512 - strm.avail_out;
        ofstream.write((const char*)temp.data(), have);
    } while (strm.avail_out == 0);
    assert(strm.avail_in == 0);     /* all input will be used */
    ::deflateEnd(&strm);
}

void gzCompress(const void* p, size_t len)
{
    auto fp = ::gzopen("replay_server_out.gz", "wb");
    auto out_len = ::gzwrite(fp, p, len);
    ::gzclose(fp);
    std::cout << "len: " << len << "   gz_len: " << out_len << "\n";
}

int main()
{
    try {
        auto t_begin = std::chrono::system_clock::now();
        fun();
        auto t_end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_begin).count() << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cout << "boost::exception " << e.what() << "\n";
    }
}
