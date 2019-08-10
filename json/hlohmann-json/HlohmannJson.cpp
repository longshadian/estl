#include <cstring>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void test()
{
	json j2 = 
	{
		{"pi", 3.141},
		{"happy", true},
		{"name", "Niels"},
		{"nothing", nullptr},
		{"answer", 
			{
				{"everything", 42}
			}
		},
		{"list", {1, "aaa", nullptr}},
		{"object", 
			{
				{"currency", "哈哈哈哈"},
				{"value", 42.99}
			}
		}
	};

	auto s = j2.dump(4);
    std::cout << s << "\n";

	try {
        auto j3 = json::parse(s);
		std::cout << j3["object"]["currency"] << "\n";
		std::cout << j3["list"][0] << "\n";
		std::cout << j3["list"][1] << "\n";
		std::cout << j3["list"][2].is_null() << "\n";
		std::cout << j3["answer"]["everything"] << "\n";
		std::cout << j3["xx"].get<std::string>() << "\n";
	} catch (const std::exception& e) {
		std::cout << "parse failed. exception: " << e.what() << "\n";
	}
}

void test2()
{
	try {
		json j = 
		{
			{ "abc", {{"ccc", 1}} }
		};
		auto v = j["abc"]["ccc"].get<int>();
		std::cout << v << "\n";
		std::cout << j["abc"]["ccc"] << "\n";
	} catch (const std::exception& e) {
		std::cout << "parse failed. exception: " << e.what() << "\n";
	}
}

void test3()
{
    std::string str = R"(
    {
        "download": [
        {
            "name": "client1009.json",
                "download_dir" : "",
                "md5" : "74A829A603CA91D4F4C3906DB7493DED",
                "url" : "http://upgradepkg.aida58.com:9091/zhwkUpdatePkg/client",
                "property" : {
                "type": "",
                    "version" : ""
            }
        },
        {
            "name": "client1009.zip",
            "download_dir" : "",
            "md5" : "4D613C97E59B79000003A31D2D9A35B5",
            "url" : "http://upgradepkg.aida58.com:9091/zhwkUpdatePkg/client",
            "property" : {
                "type": "",
                "version" : ""
            }
        },
        {
            "name": "zhwkclientupdate.json",
            "download_dir" : "",
            "md5" : "9FB22E7C33AA2FD366F8518F54E0A05C",
            "url" : "http://upgradepkg.aida58.com:9091/zhwkUpdatePkg/client",
            "property" : {
                "type": "",
                "version" : ""
            }
        }
        ]
    }
        )";

	try {
        const json& j = json::parse(str);
        const json& download = j["download"];
        for (const json& it : download) {
            std::cout << j["version"].get<std::string>() << "\n";
        }
        std::cout << download.size() << "\n";
	} catch (const std::exception& e) {
		std::cout << "parse failed. exception: " << e.what() << "\n";
	}
}

int main()
{
    system("chcp 65001");
	test3();

    return 0;
}
