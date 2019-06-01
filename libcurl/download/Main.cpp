#include <cstdio>
#include <chrono>
#include <cassert>

#include "CURLClient.h"

#include "HttpDownload.h"

std::FILE* g_f;
const std::string g_path = R"(D:\cmake_builds\libcurl_builds\)";
const std::string g_filename = "win7.iso";

std::uint64_t g_total;

void Fun(curl::CurlClient* client, std::uint64_t index, std::vector<std::uint8_t>* buffer, std::size_t length)
{
    if (!g_f)
        return;
    std::fwrite(buffer->data(), 1, buffer->size(), g_f);
    g_total += buffer->size();
    buffer->clear();
    std::fflush(g_f);
}

int TestGet()
{
    std::string to = g_path + g_filename;
    g_f = std::fopen(to.c_str(), "wb+");
    if (!g_f) {
        printf("ERROR open file %s failed.\n", to.c_str());
        return 0;
    }

    auto begin = std::chrono::steady_clock::now();

    std::string url = "http://127.0.0.1:10091/download/" + g_filename;
    curl::CurlClient client{};
    client.BindResponseCallback(std::bind(&Fun, &client, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3));
    client.Get(url);
    if (client.ResponseOK() && client.ResponseCode() == 200) {
    } else {
        printf("ERROR get failed. httpcode %d\n", client.m_http_response_code);
        return 0;
    }

    auto end = std::chrono::steady_clock::now();
    auto delta = std::chrono::duration_cast<std::chrono::seconds>(end - begin).count();
    printf("get success %s. total: %lld   time: %d\n", url.c_str(), g_total, (int)delta);

    return 0;
}

void Test3()
{
    std::string url = "http://127.0.0.1:10091/download/test/manifest.json";
    HttpDownload http{};
    std::string s = R"(D:\cmake_builds\libcurl_builds\download\xxx)";
    assert(http.Init(s));
    http.Launch(url);
}

int main()
{
    Test3();
    system("pause");
    return 0;
}

