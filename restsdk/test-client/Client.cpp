#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>

//#define _TURN_OFF_PLATFORM_STRING
#include <cpprest/http_client.h>
#include <cpprest/filestream.h>
#include <cpprest/http_listener.h>

using namespace utility;                    // Common utilities like string conversions
using namespace web;                        // Common features like URIs.
using namespace web::http;                  // Common HTTP functionality
using namespace web::http::client;          // HTTP client features
using namespace concurrency::streams;       // Asynchronous streams
using namespace web::http::experimental::listener;

using namespace std;

void TestHttpClient()
{
    auto fileStream = std::make_shared<concurrency::streams::ostream>();

    // Open stream to output file.
    pplx::task<void> requestTask = concurrency::streams::fstream::open_ostream(U("results.html"))
        .then([=](concurrency::streams::ostream outFile)
    {
        *fileStream = outFile;

        // Create http_client to send the request.
        http_client client(U("http://www.bing.com/"));

        // Build request URI and start the request.
        uri_builder builder(U("/search"));
        builder.append_query(U("q"), U("cpprestsdk github"));
        return client.request(methods::GET, builder.to_string());
    })

    // Handle response headers arriving.
    .then([=](http_response response)
    {
        printf("Received response status code:%u\n", response.status_code());

        // Write response body into the file.
        return response.body().read_to_end(fileStream->streambuf());
    })

    // Close the file stream.
    .then([=](size_t)
    {
        return fileStream->close();
    });

    // Wait for all the outstanding I/O to complete and handle any exceptions
    try
    {
        requestTask.wait();
    }
    catch (const std::exception &e)
    {
        printf("Error exception:%s\n", e.what());
    }
}

void TestHttpClient2()
{
    http_client client(U("https://www.baidu.com"));
    uri_builder builder(U("/"));
    //builder.append_query(U("q"), U("cpprestsdk github"));
    http_response response = client.request(methods::GET, builder.to_string()).get();
    printf("Received response status code:%u\n", response.status_code());

    std::cout << "client response:\n";
    ucout << response.to_string() << "\n";
}

int main(int argc, char* argv[])
{
    int n = 0;
    while (1) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        TestHttpClient2();
        ++n;
        std:cout << n << "\n\n";
    }
    return 0;
}

