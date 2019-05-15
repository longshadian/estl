
#include <iostream>
#include <vector>

#include <boost/circular_buffer.hpp>


void printF(const boost::circular_buffer<int>& buff)
{
    for (auto v : buff)
        printf("%5d,", v);
    std::cout << "\n";
}

int main()
{
    boost::circular_buffer<int> gen{5};
    {
        std::vector<int> v = { 1,2,3,4,5,6,7 };
        gen.assign(v.begin(), v.end());
    }

    /*
    while (!gen.empty()) {
        auto v = gen.front();
        std::cout << v << "\n";
        gen.pop_front();
        if (!gen.empty())
            std::cout << "    " << gen[0] << "\n";
    }
    */
    printF(gen);
    gen.pop_front();
    printF(gen);
    gen.push_back(10);
    printF(gen);
    gen.push_back(11);
    printF(gen);

    return 0;
};

