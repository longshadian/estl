#include <memory>
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "httplib.h"

class HttpServer
{
public:

    HttpServer() = default;
    ~HttpServer() = default;

    int Init(std::string host, int port)
    {
        st.Post("/name", [this](const httplib::Request& req, httplib::Response& res)
        {
            this->PostName(req, res);
        });

        st.Get("/except", [this](const httplib::Request& req, httplib::Response& res)
        {
            this->GetExcept(req, res);
        });

        st.Get("/timeout", [this](const httplib::Request& req, httplib::Response& res)
        {
            this->GetTimeout(req, res);
        });

        st.set_error_handler([this](const httplib::Request& req, httplib::Response& res) {
            auto fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
            char buf[BUFSIZ];
            snprintf(buf, sizeof(buf), fmt, res.status);
            res.set_content(buf, "text/html");
            std::cout << "error handler\n";
        });

        st.listen(host.c_str(), port);
        return 0;
    }

    void PostName(const httplib::Request& req, httplib::Response& res)
    {
        std::cout <<  "uri: body: " << req.body << "\n";
        res.set_content("success", "text/plain");
    }

    void GetExcept(const httplib::Request& req, httplib::Response& res)
    {
        std::cout <<  "except\n ";
        throw std::runtime_error("get exception");
        res.set_content("success", "text/plain");
    }

    void GetTimeout(const httplib::Request& req, httplib::Response& res)
    {
        std::cout <<  "timeout\n ";
        std::this_thread::sleep_for(std::chrono::seconds(3));
        res.set_content("success", "text/plain");
    }

    httplib::Server st;
};

int main()
{
    HttpServer hs;
    hs.Init("127.0.0.1", 18080);
    //hs.Init("localhost", 18080);
    return 0;
}

