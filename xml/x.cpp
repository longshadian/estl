
#include <tinyxml2.h>

#include <string>
#include <iostream>
#include <vector>

int main()
{
	std::string s =
	"<xml><return_code><![CDATA[SUCCESS]]></return_code>"
	"<return_msg><![CDATA[OK]]></return_msg>"
	"<appid><![CDATA[wx44260df0d471fd8e]]></appid>"
	"<mch_id><![CDATA[1378628302]]></mch_id>"
	"<nonce_str><![CDATA[9SSJrA6vCYgjG2BS]]></nonce_str>"
	"<sign><![CDATA[DFF9B9E2B6EC0A44C9E9E1955878F347]]></sign>"
	"<result_code><![CDATA[SUCCESS]]></result_code>"
	"<prepay_id><![CDATA[wx20170301112318c8db0cea450691744778]]></prepay_id>"
	"<trade_type><![CDATA[APP]]></trade_type>"
	"</xml>"

    tinyxml2::XMLDocument doc{};
	  if (tinyxml2::XML_SUCCESS != doc.Parse(xml_str.c_str())) {
		std::cout << "xml_str parse to xml fail\n";
		return 0;
    }

    //微信返回包含CDATA
    auto* xml = doc.FirstChildElement("xml");
    if (xml) {
		std::cout << "xml_str node xml null\n";
        return 0;
    }


    tinyxml2::XMLElement* appid = doc.NewElement("appid");
    appid->SetText("appid_1");
    xml->InsertEndChild(appid);

    tinyxml2::XMLElement* mch_id = doc.NewElement("mch_id");
    mch_id->SetText("appid_2");
    xml->InsertEndChild(mch_id);

    doc.InsertEndChild(xml);

    tinyxml2::XMLPrinter xml_p{};
    doc.Print(&xml_p);


    std::cout << xml_p.CStr();

    return 0;
}
