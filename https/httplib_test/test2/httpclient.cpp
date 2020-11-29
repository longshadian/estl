#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "httplib.h"

void TestTimeout(std::string host)
{
    httplib::Client cli(host.c_str());
    cli.set_read_timeout(1);
    cli.set_write_timeout(1);

    auto res = cli.Get("/timeout");
    if (res) {
        if (res->status == 200) {
            std::cout << res->body << std::endl;
        } else {
            std::cout << "error: " << res->body << "\n";
        }
    } else {
        std::cout << res.error() << "\n";
    }
}

void TestPost(std::string host)
{
    httplib::Client cli(host.c_str());
    cli.set_read_timeout(1);
    cli.set_write_timeout(1);

    std::string content = R"(
{
    "key":100,
    "value": "abc"
}
)";

    auto res = cli.Post("/name", content, "application/json;charset=utf-8");
    if (res) {
        if (res->status == 200) {
            std::cout << res->body << std::endl;
        } else {
            std::cout << "error: " << res->body << "\n";
        }
    } else {
        std::cout << "res: errno: " << res.error() << "\n";
    }
}

int main()
{
    std::string host = "http://127.0.0.1:18080";
    //TestTimeout(host);
    TestPost(host);
    return 0;
}

