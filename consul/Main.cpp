#include <iostream>
#include <string>
#include <chrono>
#include <boost/log/trivial.hpp>

#include <nlohmann/json.hpp>

#include "Types.h"
#include "HttpClient.h"
#include "Consul.h"

using namespace nlohmann;


void test()
{
    try {
        auto tbegin = std::chrono::system_clock::now();

        std::string host = "192.168.0.242";
        unsigned short port = 8500;
        //std::string target = "/v1/health/state/passing";
        std::string target = "/v1/catalog/services";

        HttpClient rest{};
        http::request<http::empty_body> req{};
        auto rsp = rest.Request(http::verb::get, "192.168.0.242", 8500, target, std::move(req));

        auto tend = std::chrono::system_clock::now();
        //LOG(debug) << "rsp: " << rsp;

        try {
            const auto& body = rsp.body();
            //X<decltype(body.data())> x;
            auto content = json::parse(body);
            LOG(debug) << "size: " << content.size();
            for (const auto& kv : content.items()) {
                LOG(debug) << "key: " << kv.key() << "\t" << kv.value().size();
            }
        } catch (std::exception& e) {
            LOG(warning) << "json exception: " << e.what();
        }

        LOG(debug) << "\n\n\n cost: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
    } catch (const std::exception& e) {
        LOG(error) << "exception: " << e.what();
    }
}

void testConsul()
{
    try {
        auto tbegin = std::chrono::system_clock::now();

        std::string host = "192.168.0.242";
        unsigned short port = 8500;
        Consul consul{};
        consul.Initialize(host, port);

        auto ret = consul.HealthStatePassing();
        auto tend = std::chrono::system_clock::now();

        LOG(debug) << "\n\n\n success: " << ret << " cost: " 
            << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();
    } catch (const std::exception& e) {
        LOG(error) << "exception: " << e.what();
    }
}

int main()
{
    testConsul();
    if (true)
        system("pause");
    return 0;
}
