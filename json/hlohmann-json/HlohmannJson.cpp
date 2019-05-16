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
				{"currency", "USD"},
				{"value", 42.99}
			}
		}
	};

	auto s = j2.dump();

	try {
		auto j3 = json::parse(s);
		std::cout << j3["object"]["currency"] << "\n";
		std::cout << j3["list"][0] << "\n";
		std::cout << j3["list"][1] << "\n";
		std::cout << j3["list"][2].is_null() << "\n";
		std::cout << j3["answer"]["everything"] << "\n";
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

int main()
{
	//test();
	test2();

    return 0;
}
