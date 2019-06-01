
#include "BoostBeast.h"
//#include "HttpSession.h"
#include "HttpServer.h"

#include <nlohmann/json.hpp>

void TestGet(http::request<http::string_body>& req, http::response<http::string_body>& resp)
{
    BOOST_LOG_TRIVIAL(debug) << req.target();
    resp.result(http::status::ok);
    resp.set(boost::beast::http::field::server, BOOST_BEAST_VERSION_STRING);
    resp.set(boost::beast::http::field::content_type, "text/html");
    resp.body() = "aaaaab";
}

int main(int argc, char* argv[])
{
    BOOST_LOG_TRIVIAL(debug) << "aaaaaaa";
    std::cerr <<
        "Usage: advanced-server-flex <address> <port> <doc_root> <threads>\n" <<
        "Example:\n" <<
        "    advanced-server-flex 0.0.0.0 8080 . 1\n";

    std::vector<HttpServer::Resource> resource = 
    {
        { "/test_get",	"GET",  std::bind(&TestGet, std::placeholders::_1, std::placeholders::_2) },
    };

    HttpServer http_server{};
    http_server.InitSslContext();
    http_server.InitResource(std::move(resource));

    http_server.Start(3);
    http_server.WaitExit();
    system("pause");
    return EXIT_SUCCESS;
}
