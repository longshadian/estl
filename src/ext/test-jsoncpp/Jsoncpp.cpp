#include <iostream>
#include <string>

#include <json/json.h>

using namespace Json;


bool StringToJson(const std::string& str, void* p)
{
    try {
        Json::Value* pjson = reinterpret_cast<Json::Value*>(p);
        Json::Value& root = *pjson;

        Json::CharReaderBuilder jsreader;
        std::unique_ptr<Json::CharReader> reader(jsreader.newCharReader());
        std::string err;
        return reader->parse(str.c_str(), str.c_str() + str.length(), &root, &err);
    }
    catch (...) {
        return false;
    }
}

bool JsonToString(const void* p, std::string& str)
{
    try {
        const Json::Value* pjson = reinterpret_cast<const Json::Value*>(p);
        const Json::Value& root = *pjson;
        Json::StreamWriterBuilder jsrocd;
        jsrocd["commentStyle"] = "None";
        jsrocd["indentation"] = " ";
        std::unique_ptr<Json::StreamWriter> writer(jsrocd.newStreamWriter());
        std::ostringstream os;
        writer->write(root, &os);
        str = os.str();
        return true;
    }
    catch (...) {
        return false;
    }
}

int Fun()
{
    //std::string s = R"({"a":1,"b":{"c":2}})";
    std::string s = R"({"a":1,"b":2.3})";
    Value v;
    Reader reader;
    if (!reader.parse(s, v)) {
        std::cout << "parse error\n";
        return 0;
    }
    std::cout << s << "\n";

    try {
        std::cout << "isMember a:" << v.isMember("a") << "\n";
        std::cout << "isMember b:" << v.isMember("b") << "\n";
        std::cout << "isMember d:" << v.isMember("d") << "\n";
        std::cout << v["a"].isNumeric() << "\n";
        std::cout << v["b"].isNumeric() << "\n";
        if (v["b"].isObject())
            std::cout << "isMember c:" << v["b"].isMember("c") << "\n";
    } catch (const std::exception& e) {
        std::cout << "excpetion:" << e.what() << "\n";
    }
    return 0;
}

int main()
{
    std::string s = R"(
        { "a":  1,
         "b"   : 2.3})";
    std::cout << s << "\n";
    Json::Value v;
    StringToJson(s, &v);
    std::cout << v.toStyledString();

    std::cout << "----\n";
    std::string str;

    JsonToString(&v, str);
    std::cout << str << "\n";

    Json::FastWriter writer;
    str = writer.write(v);
    std::cout << "-----\n";
    std::cout << str << "\n";

    return 0;
}