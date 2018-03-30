#include <iostream>
#include <chrono>
#include <thread>
#include <string>

int main()
{
	std::string s = "xxxx";

	int n = 0;
	while(true) {
		std::cout << s << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(1));
		if (++n > 10) 
			break;
	}
	return 0;
}
