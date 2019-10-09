#include "HttpServer.h"

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <thread>
#include <string>
#include <vector>

int X = 0;

using namespace boost;

std::string ThreadID()
{
    std::ostringstream ostm{};
    ostm << std::this_thread::get_id();
    return ostm.str();
}

void GetName(beast::http::request<beast::http::string_body>& req, beast::http::response<beast::http::string_body>& resp)
{
    auto s = ThreadID();
    int v = ++X;
    printf("xxxx thread:%s %d\n", s.c_str(), v);
    resp.body() = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa " + std::to_string(v);
    if (X%5 == 0) {
        //throw std::runtime_error("throw exception: " + std::to_string(X));
    }
}

void PostName(beast::http::request<beast::http::string_body>& req, beast::http::response<beast::http::string_body>& resp)
{
    std::ostringstream ostm{};
    ostm << "post success " << ++X;
    resp.body() = ostm.str();
    std::string post_content = req.body();
    if (post_content.empty()) {
        printf("post content is empty.\n");
    } else {
        printf("post content %s\n", post_content.c_str());
    }
}

int Test()
{
    bhttp::SetLogCallback([](int level, const std::string& content) { printf("%d     %s\n", level, content.c_str()); });

    auto pserver = std::make_shared<bhttp::HttpServer>();
    bhttp::HttpServer& server = *pserver;
    server.SetHost("0.0.0.0");
    server.SetPort(8080);
    if (!server.Init(1)) {
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
    bhttp::SetLogCallback(
        [](int level, const char* content) 
        { 
            std::string s = ThreadID();
            printf("[%d] [thread:%s]     %s\n", level, s.c_str(), content); 
        }
    );
    auto p = std::make_shared<bhttp::HttpServer>();
    p->SetHost(host);
    p->SetPort(port);
    p->AddHttpHandler("GET", "/name", std::bind(&GetName, std::placeholders::_1, std::placeholders::_2));
    p->AddHttpHandler("POST", "/postname", std::bind(&PostName, std::placeholders::_1, std::placeholders::_2));
    return p;
}

void Test2()
{
    auto p = CreateHttpServer("0.0.0.0", 8080);

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
        std::this_thread::sleep_for(std::chrono::seconds{1});
        //printf("sleep %d\n", n);
        if (n == 20) {
            printf("http shutdown\n");
            //p->Shutdown();
        }
        if (n == 30) {
            //break;
        }
    }
}

int main(int argc, char* argv[])
{
    Test2();
    //std::system("pause");
    return 0;
}

