#include <iostream>

#include "room.pb.h"
#include <json/json.h>
#include <google/protobuf/util/json_util.h>

int testClient(const std::string& s)
{
    Json::Value val;
    Json::Reader reader;
    if (!reader.parse(s, val)) {
        std::cout << "json error\n";
        return 0;
    }

    pt::room_client_conf client;
    auto ret = google::protobuf::util::JsonStringToMessage(s, &client);
    if (!ret.ok()) {
        std::cout << "proto error\n";
        return 0;
    }

    for (int i = 0; i != client.data_size(); ++i) {
        const auto& c = client.data(i);
        std::cout << "roomid:" << c.roomid() << "\n";
        std::cout << "min_gold:" << c.min_gold() << "\n";
        std::cout << "max_gold:" << c.max_gold() << "\n";
        std::cout << "max_round:" << c.max_round() << "\n";
        std::cout << "ante:" << c.ante() << "\n";
        std::cout << "time:" << c.time() << "\n";
        std::cout << "json_str:" << c.json_str() << "\n";
        std::cout << "===========\n";
    }
}

int testServer(const std::string& s)
{
    Json::Value val;
    Json::Reader reader;
    if (!reader.parse(s, val)) {
        std::cout << "json error\n";
        return 0;
    }

    pt::room_server_conf server;
    auto ret = google::protobuf::util::JsonStringToMessage(s, &server);
    if (!ret.ok()) {
        std::cout << "proto error\n";
        return 0;
    }

    for (int i = 0; i != server.data_size(); ++i) {
        const auto& c = server.data(i);
        std::cout << "name:" << c.name() << "\t" << c.name().size() << "\n";
        std::cout << "min_gold:" << c.min_gold() << "\n";
        std::cout << "max_gold:" << c.max_gold() << "\n";
        std::cout << "max_round:" << c.max_round() << "\n";
        std::cout << "ante:" << c.ante() << "\n";
        std::cout << "time:" << c.time() << "\n";
        std::cout << "json_str:" << c.json_str() << "\n";
        std::cout << "===========\n";
    }
}

int main()
{
    std::string cstr = R"({"data":[{"ante": 9, "time" : "2017-02-07 00:00:00", "roomid" : 1001, "min_gold" : 1, "json_str" : "{ \"name\": \"BeJson\", \"url\": \"son.com\"}", "max_gold" : 100, "max_round" : 10}, { "ante": 8, "time" : "2017-02-03 00:00:00", "roomid" : 1002, "min_gold" : 2, "max_gold" : 200, "max_round" : 20 }, { "ante": 7, "time" : "2017-02-01 00:00:00", "roomid" : 1003, "min_gold" : 3, "max_gold" : 300, "max_round" : 30 }]})";
    std::string sstr = R"({"data":[{"ante": 9, "time": "2017-02-07 00:00:00", "min_gold": 1, "json_str": "{ \"name\": \"BeJson\", \"url\": \"son.com\"}", "max_gold": 100, "max_round": 10, "name": "\u521d\u7ea7\u573a"}, {"max_gold": 200, "max_round": 20, "time": "2017-02-03 00:00:00", "ante": 8, "min_gold": 2}, {"ante": 7, "time": "2017-02-01 00:00:00", "min_gold": 3, "max_gold": 300, "max_round": 30, "name": "\u9ad8\u7ea7\u573a"}]})";
    testServer(sstr);
    return 0;
}

