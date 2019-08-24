#include "HttpServer.h"

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

int X = 0;

using namespace boost;

void GetName(beast::http::request<beast::http::string_body>& req, beast::http::response<beast::http::string_body>& resp)
{
    printf("xxxxxxxxxxxxxx %d\n", ++X);
    resp.body() = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa " ;
}

void PostName(beast::http::request<beast::http::string_body>& req, beast::http::response<beast::http::string_body>& resp)
{

}

int Test()
{
    bhttp::SetLogFunc([](int level, const std::string& content) { printf("%d     %s\n", level, content.c_str()); });

    auto pserver = std::make_shared<bhttp::HttpServer>();
    bhttp::HttpServer& server = *pserver;
    //server.SetHost("127.0.0.1");
    server.SetHost("192.168.97.15");
    server.SetPort(6079);

    if (!server.Init(3)) {
        printf("server init failed.");
        return 0;
    }

    server.AddHttpHandler("GET", "/name", std::bind(&GetName, std::placeholders::_1, std::placeholders::_2));

    int n = 0;
    while (1) {
        ++n;
        std::this_thread::sleep_for(std::chrono::seconds{1});
        printf("sleep %d\n", n);

        //if (n == 6) break;
    }

    server.Shutdown();
    pserver = nullptr;

    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds{1});
        printf("after shutdown");
    }
    return 0;
}


std::shared_ptr<bhttp::HttpServer> CreateHttpServer(std::string host, uint16_t port)
{
    bhttp::SetLogFunc([](int level, const std::string& content) { printf("%d     %s\n", level, content.c_str()); });
    auto p = std::make_shared<bhttp::HttpServer>();
    p->SetHost(host);
    p->SetPort(port);
    p->AddHttpHandler("GET", "/name", std::bind(&GetName, std::placeholders::_1, std::placeholders::_2));
    return p;
}

void Test2()
{
    auto p = CreateHttpServer("192.168.97.15", 6079);

    bool init_succ = false;
    do {
        if (p->Init(3)) {
            init_succ = true;
        } else {
            printf("http server init failed.\n");
            std::this_thread::sleep_for(std::chrono::seconds{3});
            p->Reset();
            p->AddHttpHandler("GET", "/name", std::bind(&GetName, std::placeholders::_1, std::placeholders::_2));
        }
    } while (!init_succ);

    printf("http server init success ----------------\n");

    int n = 0;
    while (1) {
        ++n;
        std::this_thread::sleep_for(std::chrono::seconds{3});
        printf("sleep %d\n", n);
        //if (n == 6) break;
    }
}

int main(int argc, char* argv[])
{
    Test2();
    std::system("pause");
    return 0;
}

