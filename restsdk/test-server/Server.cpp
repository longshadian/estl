#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

//#define _TURN_OFF_PLATFORM_STRING
#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <cpprest/http_listener.h>

#if defined(_MSC_VER)
    //#include <pplx/threadpool.h>
#else
    #include <pplx/threadpool.h>
#endif


using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams
using namespace web::http::experimental::listener;

using namespace std;

class HttpServer;
std::shared_ptr<HttpServer> g_http_server;

struct ProxyInfo
{
    utility::string_t proxy_uri_;
    utility::string_t proxy_host_;
};

class HttpServer
{
public:
    HttpServer();
    HttpServer(utility::string_t url);
    virtual ~HttpServer();

    pplx::task<void>open() { return m_listener.open(); }
    pplx::task<void>close() { return m_listener.close(); }

protected:

private:
    void handle_get(http_request message);
    void handle_put(http_request message);
    void handle_post(http_request message);
    void handle_delete(http_request message);
    void handle_error(pplx::task<void>& t);

    void handle_get_sync(http_request message);
    void handle_get_async_file(http_request message);
    void handle_get_async_buffer(http_request message);
    void handle_get3(http_request message);

    http_listener m_listener;
};

HttpServer::HttpServer()
{
    //ctor
}
HttpServer::HttpServer(utility::string_t url) :m_listener(url)
{
    m_listener.support(methods::GET, std::bind(&HttpServer::handle_get_async_buffer, this, std::placeholders::_1));
    m_listener.support(methods::PUT, std::bind(&HttpServer::handle_put, this, std::placeholders::_1));
    m_listener.support(methods::POST, std::bind(&HttpServer::handle_post, this, std::placeholders::_1));
    m_listener.support(methods::DEL, std::bind(&HttpServer::handle_delete, this, std::placeholders::_1));

}
HttpServer::~HttpServer()
{
    //dtor
}

void HttpServer::handle_error(pplx::task<void>& t)
{
    try
    {
        t.get();
    }
    catch (const std::exception& e)
    {
        ucout << U("handle exception: ") << e.what() << "\n";
        // Ignore the error, Log it if a logger is available
    }
}

void HttpServer::handle_get(http_request message)
{
    //ucout << message.to_string() << endl;
    //auto paths = http::uri::split_path(http::uri::decode(message.relative_uri().path()));
    utf8string rep = "get";
    message.reply(status_codes::OK, rep)
        .then([](pplx::task<void> t)
        {
            try {
                t.get();
            } catch (...) {
            }
        });

#if 0
    message.relative_uri().path();
    //Dbms* d  = new Dbms();
    //d->connect();

    concurrency::streams::fstream::open_istream(U("static/index.html"), std::ios::in).then([=](concurrency::streams::istream is)
        {
            message.reply(status_codes::OK, is, U("text/html"))
                .then([](pplx::task<void> t)
                    {
                        try {
                            t.get();
                        }
                        catch (...) {
                            //
                        }
                    });
        }).then([=](pplx::task<void>t)
            {
                try {
                    t.get();
                }
                catch (...) {
                    message.reply(status_codes::InternalError, U("INTERNAL ERROR "));
                }
            });

        return;
 #endif

}

void HttpServer::handle_get_sync(http_request message)
{
    ucout << "absolute_uri: " << message.absolute_uri().to_string() << "\n";
    ucout << "method: " << message.method() << "\n";
    ucout << "version: " << message.http_version().to_utf8string().c_str() << "\n";

    ucout << "scheme: " << message.absolute_uri().scheme() << "\n";
    ucout << "host: " << message.absolute_uri().host() << "\n";
    ucout << "port: " << message.absolute_uri().port() << "\n";

    if (1) {
        //http_client client(U("http://www.bing.com"));
        http_client client(U("http://purecpp.org"));
        http_response response = client.request(message).get();
        printf("Received response status code:%u\n", response.status_code());

        ucout << response.to_string() << "\n";

        auto paths = http::uri::split_path(http::uri::decode(message.relative_uri().path()));
        message.reply(response);
    } else {
        message.reply(status_codes::OK, "ok");
    }
};

void HttpServer::handle_get_async_file(http_request message)
{
    auto fileStream = std::make_shared<concurrency::streams::ostream>();
    //pplx::task<void> requestTask = concurrency::streams::fstream::open_ostream(U("results.html"))
    concurrency::streams::fstream::open_ostream(U("results.html"))
        .then([=](concurrency::streams::ostream outFile)
        {
            *fileStream = outFile;

            std::vector<ProxyInfo> vec =
            {
                {U("https://www.baidu.com"),       U("www.baidu.com")},
                {U("https://cn.bing.com"),         U("cn.bing.com")},
                {U("http://purecpp.org"),          U("purecpp.org")},
            };
            const ProxyInfo info = vec[1];
            ucout << "proxy_uri: " << info.proxy_uri_ << " host: " << info.proxy_host_ << "\n";
            http_client client(info.proxy_uri_);
            http_request upstream(message.method());
            upstream.set_request_uri(message.request_uri());
            upstream.headers() = message.headers();
            upstream.headers().remove(U("Host"));
            upstream.headers().add(U("Host"), info.proxy_host_);
            return client.request(upstream);
        })

        .then([=](http_response response)
        {
                ucout << "Received response status code: " << response.status_code() << "\n";
                message.reply(response);
                //return response.body().read_to_end(fileStream->streambuf());
        })

        .then([=](pplx::task<void> t)
        {
            try {
                t.wait();
            } catch (const std::exception & e) {
                ucout << "Error exception: " << e.what() << "\n";
            }
        })
    ;
    /*
    .then([=](size_t)
        {
            return fileStream->close();
        });
    */
};

void HttpServer::handle_get_async_buffer(http_request message)
{
    pplx::create_task([=]()
        {
            std::vector<ProxyInfo> vec =
            {
                {U("https://www.baidu.com"),       U("www.baidu.com")},
                {U("https://cn.bing.com"),         U("cn.bing.com")},
                {U("http://purecpp.org"),          U("purecpp.org")},
            };
            const ProxyInfo info = vec[0];
            ucout << "proxy_uri: " << info.proxy_uri_ << " host: " << info.proxy_host_ << "\n";
            http_client client(info.proxy_uri_);
            http_request upstream(message.method());
            upstream.set_request_uri(message.request_uri());
            upstream.headers() = message.headers();
            upstream.headers().remove(U("Host"));
            upstream.headers().add(U("Host"), info.proxy_host_);
            return client.request(upstream);
        })

        .then([=](http_response response)
        {
                ucout << "Received response status code: " << response.status_code() << "\n";
                message.reply(response);
                //return response.body().read_to_end(fileStream->streambuf());
        })

        .then([=](pplx::task<void> t)
        {
            try {
                t.wait();
            } catch (const std::exception & e) {
                ucout << "Error exception: " << e.what() << "\n";
                message.reply(web::http::status_codes::InternalError);
            }
        })
    ;
};

void HttpServer::handle_get3(http_request message)
{
/*
    ucout << "absolute_uri: " << message.absolute_uri().to_string() << "\n";
    ucout << "method: " << message.method() << "\n";
    ucout << "version: " << message.http_version().to_utf8string().c_str() << "\n";

    ucout << "scheme: " << message.absolute_uri().scheme() << "\n";
    ucout << "host: " << message.absolute_uri().host() << "\n";
    ucout << "port: " << message.absolute_uri().port() << "\n";
    */
    //ucout << message.to_string() << "\n";

    auto fileStream = std::make_shared<concurrency::streams::ostream>();
    // Open stream to output file.
    pplx::task<void> requestTask = concurrency::streams::fstream::open_ostream(U("results.html"))
    .then([=](concurrency::streams::ostream outFile)
    {
        *fileStream = outFile;
        utility::string_t host1;
        utility::string_t host2;
        int i = 1;
        if (i == 0) {
            host1 = U("https://www.baidu.com");
            host2 = U("www.baidu.com");
        } else if (i == 1) {
            host1 = U("https://cn.bing.com");
            host2 = U("cn.bing.com");
        } else if (i == 2) {
            host1 = U("http://purecpp.org");
            host2 = U("purecpp.org");
        }

        http_client client(host1);
        http_request upstream(message.method());
        const auto& req_uri = message.request_uri(); 
        ucout << req_uri.to_string() << "\n";
        upstream.set_request_uri(req_uri);
        upstream.headers() = message.headers();
        upstream.headers().remove(U("Host"));
        upstream.headers().add(U("Host"), host2);

        //upstream.headers().remove(U("Accept-Encoding"));
        //upstream.headers().add(U("Accept-Encoding"), U("gzip, deflate, br"));
        if (upstream.method() == U("GET") || upstream.method() == U("HEAD")) {
        } else {
            upstream.set_body(message.body());
        }
        return client.request(upstream);
    })

    // Handle response headers arriving.
    .then([=](http_response response)
        {
            ucout << "Received response status code:" << response.status_code() << "\n";
            //ucout << "\n===========>respone:\n";
            //ucout << response.body() << "\n";
            // Write response body into the file.
            message.reply(response);
            //return response.body().read_to_end(fileStream->streambuf());
        })
        ;
        /*
    .then([=](size_t)
        {
            return fileStream->close();
        });
        */

    // Wait for all the outstanding I/O to complete and handle any exceptions
    try
    {
        requestTask.wait();
    }
    catch (const std::exception & e)
    {
        ucout << "Error exception: " << e.what() << "\n";
    }
};

void HttpServer::handle_post(http_request message)
{
    //ucout << message.to_string() << endl;
    std::string rep = "post";
    message.reply(status_codes::OK, rep);
    return;
};

void HttpServer::handle_delete(http_request message)
{
    //ucout << message.to_string() << endl;
    string rep = "delete";
    message.reply(status_codes::OK, rep);
};

void HttpServer::handle_put(http_request message)
{
    //ucout << message.to_string() << endl;
    string rep = "put";
    message.reply(status_codes::OK, rep);
};

#if defined (_MSC_VER)
int TestHttpServer(utility::string_t uri_str)
#else
int TestHttpServer(std::string uri_str)
#endif
{
    try {
#if defined(_MSC_VER)
        //crossplat::threadpool::initialize_with_threads(10);
#else
        crossplat::threadpool::initialize_with_threads(10);
#endif

        uri_builder ub(uri_str);
        auto addr = ub.to_uri().to_string();
        g_http_server = std::make_shared<HttpServer>(addr);
        g_http_server->open().wait();
        return 0;
    } catch (const std::exception & e) {
        std::cout << "http server exception: " << e.what() << std::endl;
        return -1;
    }
}

int main(int argc, char* argv[])
{
#if defined (_MSC_VER)
    utility::string_t host_port(U("http://127.0.0.1:11000"));
    if (TestHttpServer(host_port) != 0) {
#else
    std::string host_port = "http://0.0.0.0:11000";
    if (argc >= 2) {
        host_port = argv[1];
    }
    std::cout << "listen: " << host_port << "\n";
    if (TestHttpServer(host_port) != 0) {
#endif
        std::cout << "server init failed.\n";
        return -1;
    }

    int n = 0;
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        ++n;
        if (n % 10 == 0) {
            std::cout << n << "\n";
        }
    }
    g_http_server->close();
    return 0;
}
