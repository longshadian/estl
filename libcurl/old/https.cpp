#include <cstdio>

#include <curl/curl.h>
#include "HttpClient.h"

std::string printS(CURLcode c)
{
    const char* s = ::curl_easy_strerror(c);
    if (s)
        return s;
    return "";
}

int main()
{
    ::curl_global_init(CURL_GLOBAL_DEFAULT);

    HttpClinet c{};
    c.m_write_cb = &HttpClinet::onWrite;
    c.m_write_cb_args = &c;

    ::curl_easy_setopt(c.m_curl, CURLOPT_SSL_VERIFYPEER, 0L);
    ::curl_easy_setopt(c.m_curl, CURLOPT_SSL_VERIFYHOST, 0L);
    //::curl_easy_setopt(c.m_curl, CURLOPT_CAPATH, "/home/cgy/work/test/libcurl/server.crt");

    //c.post("https://127.0.0.1:22001/info", "qwert123456");
    c.post("https://127.0.0.1/info", "qwert123456");
    if (!c.responseOK()) {
        std::cout << "post error! " << c.m_error;
        std::cout << printS(c.m_error) << "\n";
        ::curl_global_cleanup();
        return 0;
    }
    std::cout << c.m_response << "\n";
    std::cout << printS(c.m_error) << "\n";
    ::curl_global_cleanup();
    return 0;
}
