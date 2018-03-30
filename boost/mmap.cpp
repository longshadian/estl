

#include <cstdio>

#include <iostream>
#include <memory>
#include <chrono>
#include <string>
#include <vector>

struct FileClose
{
    FileClose(const std::string& path)
        : m_file(nullptr)
    {
        m_file = ::fopen(path.c_str(), "a+");
    }

    ~FileClose()
    {
        if (m_file)
            ::fclose(m_file);
    }

    operator bool() const
    {
        return m_file != nullptr;
    }

    FILE* m_file;
};

void testOpen()
{
    std::string FILE_NAME = "test/x.bin.";
    std::vector<std::shared_ptr<FileClose>> all_files{};

    std::chrono::system_clock::time_point tnow = std::chrono::system_clock::now();
    int total = 100;
    for (int i = 0; i != total; ++i) {
        auto fs = std::make_shared<FileClose>(FILE_NAME + std::to_string(i));
        if (*fs)
            all_files.push_back(fs);
    }

    std::chrono::system_clock::time_point tend = std::chrono::system_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(tend - tnow);
    std::cout << "open " << total << " cost: " << delta.count() << "\n";
}

void testWrite()
{
    std::string FILE_NAME = "test/w.bin";
    auto fs = std::make_shared<FileClose>(FILE_NAME);

    size_t str_len = 200;
    std::vector<uint8_t> str{};
    str.resize(str_len, 'a');

    std::chrono::system_clock::time_point tnow = std::chrono::system_clock::now();
    int count = 1000;
    for (int i = 0; i != count; ++i) {
        ::fwrite(str.data(), str.size(), 1, fs->m_file);
    }

    std::chrono::system_clock::time_point tend = std::chrono::system_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::microseconds>(tend - tnow);
    std::cout << "write len: " << str_len 
        << " count: " << count 
        << " cost: " << delta.count() << "\n";
}

int main()
{
    //testOpen();
    testWrite();
    return 0;
}
