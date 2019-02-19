
#include <tinyxml2.h>

#include <string>
#include <iostream>
#include <vector>

int main()
{
    const char* TEXT =
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

    tinyxml2::XMLDocument doc{};
    doc.Parse(TEXT);
    doc.Print();

    std::cout << doc.ErrorID() << "\n";

    auto s1 = doc.FirstChild()->FirstChildElement("return_code")->GetText();
    std::cout << s1 << "\n";

    auto* node = doc.FirstChild();
    tinyxml2::XMLElement* e_fun = doc.NewElement("fun");
    //e_fun->SetName("a");
    e_fun->SetText("xxxx");
    node->InsertEndChild(e_fun);

    tinyxml2::XMLPrinter xml_p{};
    doc.Print(&xml_p);

    std::cout << "str:\n";
    std::cout << xml_p.CStr();

    return 0;

}
