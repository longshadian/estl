#include <iostream>
#include <string>
#include <string_view>
#include <cassert>
#include <sstream>

#include <enoki/array.h>
#include <enoki/matrix.h>

using Float = float;
using Vector3f = enoki::Array<Float, 3>;
using Matrix3f = enoki::Matrix<Float, 3>;

/// Generic string conversion routine
template <typename T> inline std::string to_string(const T& value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

void Test()
{
    Matrix3f m(
        1, 2, 3,
        4,5,6,
        7,8,9);
    Vector3f v(1,2,3);
    std::cout << to_string(m) << "\n";
    std::cout << to_string(v) << "\n";
    std::cout << to_string(m*v) << "\n";
    v[3];

    printf("%f %f %f\n", v[0], v[1], v[2]);
}

int main()
{
    if (1) {
        Test();
    }
    system("pause");
    return 0;
}
