
#include <boost/property_tree/ptree.hpp>
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

}

int main()
{
    boost::property_tree::ptree pt;
    pt.put("conf.a", "heheh");
    pt.put("conf.b", "xxx");

    boost::property_tree::write_xml("out.xml", pt);

    return 0;
}