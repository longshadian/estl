#include "server_http.hpp"

#include <iostream>

using SWHttpServer      = ::SimpleWeb::Server<::SimpleWeb::HTTP>;
using RequestPtr = std::shared_ptr<SWHttpServer::Request>;
using ResponsePtr = std::shared_ptr<SWHttpServer::Response>;

void send200(ResponsePtr response, const std::string& s)
{
    *response << "HTTP/1.0 200 OK\r\n"
        << "Content-Length: " << s.size() << "\r\n"
        << "\r\n" << s;
}

void defaultPost(ResponsePtr response, RequestPtr request)
{
    (void)request;
    std::string content =
        "<HTML><TITLE>Not Found</TITLE>"
        "<BODY><P>The server could not fulfill<br>"
        "your request because the resource specified is unavailable or nonexistent."
        "</BODY></HTML>";
    *response << "HTTP/1.0 404 NOT FOUND\r\n"
        << "Content-Length: " << content.length() << "\r\n"
        << "\r\n" << content;
}

void defaultGet(ResponsePtr response, RequestPtr request)
{
    std::cout << "defaultGet\n";
    defaultPost(response, request);
}

void postInfo(ResponsePtr response, RequestPtr request)
{
    auto val = request->content.size();
    std::string s = request->content.string();
    std::cout << "val " << s << std::endl;

    std::string real_ip{};
    auto it = request->header.find("X-real-ip");
    if (it != request->header.end())
        real_ip = it->second;
    std::cout << "real_ip:" << real_ip  << "\n";

    std::string real_ip2{};
    auto it2 = request->header.find("X-Forwarded-For");
    if (it2 != request->header.end())
        real_ip2 = it2->second;
    std::cout << "real_ip2:" << real_ip2 << "\n";

    std::cout << "remote ip:" << request->remote_endpoint_address << "\n";

    /*
    std::this_thread::sleep_for(std::chrono::seconds(3));
    std::stringstream content_stream;
    content_stream << "<h1>Request from " << request->remote_endpoint_address << " (" << request->remote_endpoint_port << ")</h1>";
    content_stream << request->method << " " << request->path << " HTTP/" << request->http_version << "<br>";
    for (auto& header : request->header) {
        content_stream << header.first << ": " << header.second << "<br>";
    }
    content_stream.seekp(0, std::ios::end);
    send200(response, content_stream.str());
    */
    send200(response, s);
}

void getInfo(ResponsePtr response, RequestPtr request)
{
    std::string real_ip{};
    auto it = request->header.find("X-real-ip");
    if (it != request->header.end())
        real_ip = it->second;
    std::cout << "real_ip:" << real_ip  << "\n";

    std::string real_ip2{};
    auto it2 = request->header.find("X-Forwarded-For");
    if (it2 != request->header.end())
        real_ip2 = it2->second;
    std::cout << "real_ip2:" << real_ip2 << "\n";
    std::cout << "remote ip:" << request->remote_endpoint_address << "\n";

    send200(response, std::to_string(std::time(nullptr)));
}

int main()
{
    SWHttpServer s{22001, 1};
    s.default_resource["POST"] = std::bind(&defaultPost, std::placeholders::_1, std::placeholders::_2);
    s.default_resource["GET"] = std::bind(&defaultGet, std::placeholders::_1, std::placeholders::_2);
    s.resource["^/info$"]["POST"] = std::bind(&postInfo, std::placeholders::_1, std::placeholders::_2);
    s.resource["^/info$"]["GET"] = std::bind(&getInfo, std::placeholders::_1, std::placeholders::_2);
    s.start();
    return 0;
}
