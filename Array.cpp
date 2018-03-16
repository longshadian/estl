
#include <cstdint>
#include <memory>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>

template <typename T> 
class Basic_Site_Array
{
public:
    using Array = std::vector<T>;
    using value_type = typename Array::value_type;
    using size_type = typename Array::size_type;
    using difference_type = typename Array::difference_type;
    using pointer = typename Array::pointer;
    using const_pointer = typename Array::const_pointer;
    using reference = typename Array::reference;
    using const_reference = typename Array::const_reference;
    using iterator = typename Array::iterator;
    using const_iterator = typename Array::const_iterator;
    using reverse_iterator = typename Array::reverse_iterator;
    using const_reverse_iterator = typename Array::const_reverse_iterator;

    Basic_Site_Array(size_t num)
        : m_vec()
    {
        m_vec.resize(num);
    }
    ~Basic_Site_Array() = default;
    Basic_Site_Array(const Basic_Site_Array& rhs) = default;
    Basic_Site_Array& operator=(const Basic_Site_Array& rhs) = default;

    Basic_Site_Array(Basic_Site_Array&& rhs) = default;
    Basic_Site_Array& operator=(Basic_Site_Array&& rhs) = default;

    void fill(const T& val)
    {
        for (auto& it : m_vec)
            it = val;
    }

    T& operator[](size_type pos) { return m_vec[pos]; }
    const T& operator[](size_type pos) const { return m_vec[pos]; }

    iterator begin() { return m_vec.begin(); }
    iterator end() { return m_vec.end(); }

    const_iterator begin() const { return m_vec.cbegin(); }
    const_iterator end() const { return m_vec.cend(); }

    const_iterator cbegin() const { return m_vec.cbegin(); }
    const_iterator cend() const { return m_vec.cend(); }

private:
    Array m_vec;
};

template <typename T>
using Site_Array = Basic_Site_Array<T>;

int main()
{
    Site_Array<int32_t> v{ 12 };

    v.fill(123);

    auto cnt = std::count(v.begin(), v.end(), 123);
    std::cout << cnt << "\n";

    std::cout << v[int32_t(2)] << "\n";
    return 0;
}
