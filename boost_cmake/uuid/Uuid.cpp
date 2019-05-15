
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <boost/lexical_cast.hpp>

#include <iostream>
#include <string>

int main()
{
    boost::uuids::uuid u = boost::uuids::nil_uuid();
    std::cout << u.is_nil() << "\n";

    boost::uuids::random_generator rand_gen;
    auto ru = rand_gen();
    auto s = boost::lexical_cast<std::string>(ru);
    std::cout << s << "\n";
    std::cout << ru << "\n";
    std::cout << s.size() << "\n";
    return 0;
}
