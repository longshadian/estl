#include "SHttpServer.h"

#include <iostream>

#define  SHTTP_COUT std::cout << "shttp: "

using namespace httplib;

#if 0
static std::string dump_headers(const Headers& headers) {
    std::string s;
    char buf[BUFSIZ];

    for (const auto& x : headers) {
        snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
        s += buf;
    }

    return s;
}
#endif

#if 0
static std::string dump_multipart_files(const MultipartFormDataMap& files) {
    std::string s;
    char buf[BUFSIZ];

    s += "--------------------------------\n";

    for (const auto& x : files) {
        const auto& name = x.first;
        const auto& file = x.second;

        snprintf(buf, sizeof(buf), "name: %s %s\n", name.c_str(), file.content.c_str());
        s += buf;

        snprintf(buf, sizeof(buf), "filename: %s\n", file.filename.c_str());
        s += buf;

        snprintf(buf, sizeof(buf), "content type: %s\n", file.content_type.c_str());
        s += buf;

        snprintf(buf, sizeof(buf), "text length: %zu\n", file.content.size());
        s += buf;

        s += "----------------\n";
    }

    return s;
}
#endif

SHttpServer::SHttpServer()
    : server_(std::make_shared<httplib::Server>())
    , thd_()
    , ip_("0.0.0.0")
    , port_(5555)
{
}

SHttpServer::~SHttpServer()
{
    Stop();
    if (thd_.joinable())
        thd_.join();
}

int SHttpServer::Init(const std::string& ip, int port)
{
    ip_ = ip;
    port_ = port;

    server_->Post("/upload", [this](const Request& req, Response& res) {
        std::string res_content = R"({"status": "400"})";
        res.set_content(res_content, "text/plain");
    });

    server_->Post("/cgi-bin/cgi.cgi", [this](const Request& req, Response& res) {
        SHTTP_COUT << "req: " << req.body << "\n";
        std::string res_content = R"({"status": "400"})";
        res.set_content(res_content, "text/plain");
    });

    return server_->bind_to_port(ip_.c_str(), port) ? 0 : -1;
}

void SHttpServer::Loop()
{
    server_->listen_after_bind();
}

void SHttpServer::LoopInBackground()
{
    std::thread temp_thd(std::bind(&SHttpServer::Loop, this));
    thd_ = std::move(temp_thd);
}

void SHttpServer::Stop()
{
    server_->stop();
}


