#pragma once

#include <fstream>
#include <iostream>
#include <string>

class file_utils
{
public:
    file_utils() : fstm_() { }
    ~file_utils() { }

    bool open(const char* file_name, int flag = std::ios::out | std::ios::binary, std::string* err_msg = nullptr)
    {
        try {
            fstm_.open(file_name, flag);
            return fstm_.is_open();
        } catch (const std::exception& e) {
            if (err_msg)
                *err_msg = e.what();
            return false;
        }
    }

    bool eof() const
    {
        return fstm_.eof();
    }

    void close()
    {
        fstm_.close();
    }

    std::streamsize read(void* buf, size_t len)
    {
        fstm_.read(static_cast<char*>(buf), len);
        return fstm_.gcount();
    }

    template <std::size_t N = 1024>
    std::streamsize read()
    {
        char buf[N];
        return read(buf, N);
    }

    bool good() const
    {
        return fstm_.good();
    }

    void write(const void* data, size_t len)
    {
        try {
            fstm_.write(static_cast<const char*>(data), len);
        } catch (const std::exception& e) {
            (void)e;
        }
    }

    static bool read_file(const std::string& file_name, std::string& content, std::string* err_msg = nullptr)
    {
        try {
            std::ifstream ifsm(file_name.c_str(), std::ios::binary);
            if (!ifsm)
                return false;
            ifsm.seekg(0, std::ios::end);
            auto len = ifsm.tellg();
            if (len == 0) {
                ifsm.close();
                return true;
            }
            ifsm.seekg(0, std::ios::beg);
            content.resize(len);
            ifsm.read(&content[0], len);
            ifsm.close();
            return true;
        } catch (const std::exception& e) {
            if (err_msg)
                *err_msg = e.what();
            return false;
        }
    }

    static bool write_file(const std::string& file_name, const void* data, size_t len, std::string* err_msg = nullptr)
    {
        try {
            std::ofstream ofsm(file_name, std::ios::binary | std::ios::trunc);
            if (!ofsm)
                return false;
            ofsm.write(static_cast<const char*>(data), len);
            ofsm.flush();
            ofsm.close();
            return true;
        } catch (const std::exception& e) {
            if (err_msg)
                *err_msg = e.what();
            return false;
        }
    }

    std::fstream fstm_;
};

