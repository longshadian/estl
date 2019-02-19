#include <vector>
#include <map>
#include <unordered_map>
#include <websocketpp/server.hpp>

int main()
{
    std::vector<websocketpp::connection_hdl> v{};
    std::map<websocketpp::connection_hdl, int> m{};
    //std::unordered_map<websocketpp::connection_hdl, int> m2{};

    //auto it = m.find(websocketpp::connection_hdl{});
    return 0;
}