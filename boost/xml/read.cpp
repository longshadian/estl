#include <tinyxml2.h>

#include <boost/property_tree/xml_parser.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <map>

namespace xmltool {

const std::string XMLATTR = "<xmlattr>";
const std::string XMLCOMMENT = "<xmlcomment>";
const std::string XMLATTR_DOT = "<xmlattr>.";
const std::string XMLCOMMENT_DOT = "<xmlcomment>.";


//获取子节点列表
auto getChildOptional(const boost::property_tree::ptree& node, const std::string& key) -> decltype(node.get_child_optional(key))
{
    return node.get_child_optional(key);
}

template<typename R>
boost::optional<R> getAttributeOptional(const boost::property_tree::ptree& node, const std::string& attr_name)
{
    return node.get_optional<R>(XMLATTR_DOT + attr_name);
}

template<typename R>
R getValueOptional(const boost::property_tree::ptree& node, const std::string& attr_name)
{
    return node.get_value_optional<R>(attr_name);
}

template<typename R>
R getValue(const boost::property_tree::ptree& node, const std::string& attr_name)
{
    return node.get<R>(attr_name);
}

}

void funTool()
{
    try {
        boost::property_tree::ptree pt{};
        boost::property_tree::read_xml("ddz_conf.xml", pt,
            boost::property_tree::xml_parser::trim_whitespace |
            boost::property_tree::xml_parser::no_comments);
        auto conf = pt.get_child("conf");
        auto redis_cluster = conf.get_child("redis_cluster");
        for (auto slot : redis_cluster) {
            auto redis = slot.second;
            std::cout <<"------------\n";
            if (slot.first == "redis") {
                std::cout << "id:\t" << redis.get<std::string>("<xmlattr>.id") << "\n";
                std::cout << "name:\t" << redis.get<std::string>("<xmlattr>.name") << "\n";
                std::cout << "ip:\t" << redis.get<std::string>("ip") << "\n";
                std::cout << "port:\t" << redis.get<int>("port") << "\n";
                std::cout << "value:\t" << redis.get_value<std::string>("jjjj") << "\n";

                //auto evalsha_list = redis.get_child_optional("evalsha");
                auto evalsha_list = xmltool::getChildOptional(redis, "evalsha");
                if (evalsha_list) {
                    for (auto evalsha_slot : *evalsha_list) {
                        if (evalsha_slot.first == "file") {
                            auto evalsha = evalsha_slot.second;
                            std::cout << "\t" << "id:" << evalsha.get<std::string>("<xmlattr>.id") 
                                << "\t" << evalsha.get_value<std::string>() << "\n";
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "exception:" << e.what() << "\n";
    }
}

void fun()
{
    try {
        boost::property_tree::ptree pt{};
        boost::property_tree::read_xml("script.xml", pt,
            boost::property_tree::xml_parser::trim_whitespace |
            boost::property_tree::xml_parser::no_comments);
        auto conf = pt.get_child("conf");
        auto redis_cluster = conf.get_child("evalsha");
        for (auto slot : redis_cluster) {
            auto redis = slot.second;
            std::cout <<"------------\n";
            if (slot.first == "redis") {
                std::cout << "id:\t" << redis.get<std::string>("<xmlattr>.id") << "\n";
                for (auto it : redis) {
                    if (it.first == "file") {
                        auto evalsha = it.second;
                        std::cout << "\t" << "id:" << evalsha.get<std::string>("<xmlattr>.id") 
                            << "\t" << evalsha.get_value<std::string>() << "\n";
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cout << "exception:" << e.what() << "\n";
    }
}

int main()
{
    fun();
    return 0;
}