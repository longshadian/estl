#include <boost/property_tree/detail/rapidxml.hpp>

#include <string>
#include <iostream>
#include <vector>

int main()
{
    const std::string TEXT = 
        "<xml>"
        "<return_code><![CDATA[SUCCESS]]></return_code>"
        "<return_msg>OK</return_msg>"
        "<appid><![CDATA[wx44260df0d471fd8e]]></appid>"
        "<mch_id><![CDATA[1378628302]]></mch_id>"
        "<nonce_str><![CDATA[KuZj2xzUdC4h6APR]]></nonce_str>"
        "<sign><![CDATA[D41155A48BF9456A0E20C6FFC2A52AA3]]></sign>"
        "<result_code><![CDATA[SUCCESS]]></result_code>"
        "<prepay_id><![CDATA[wx20161220185205c5b8b73d680635972279]]></prepay_id>"
        "<trade_type><![CDATA[APP]]></trade_type>"
        "</xml>";
    std::vector<char> buff{};
    buff.resize(TEXT.size()+1);
    std::copy(TEXT.begin(), TEXT.end(), buff.begin());

    boost::property_tree::detail::rapidxml::xml_document<> xml_doc;
    xml_doc.parse<0>(buff.data());
    auto* node = xml_doc.first_node();
    if (!node)
        return 0;
    std::cout << node->name() << "\n";


    std::vector<std::string> all_node = 
    {
        "return_code",
        "return_msg",
        "appid",
        "mch_id",
        "nonce_str",
        "sign",
        "result_code",
        "prepay_id",
        "trade_type",
    };
    
    for (const auto& node_name : all_node) {
        auto* next_node = node->first_node(node_name.c_str());
        if (next_node) {
            auto* nn_node = next_node->first_node();
            if (nn_node) {
                std::cout << node_name << ":" << nn_node->value() << "\n";
            } else {
                std::cout << node_name << ":" << next_node->value() << "\n";
            }
        }
    }

    //std::cout << xml_doc.first_node("")->name() << "\n";
    return 0;
}
