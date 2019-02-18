#include "MinMaxHeap.h"

#include <iostream>
#include <vector>

class Pic
{
public:
    Pic(int v) : m_x(v), m_y(v * v) {}
    ~Pic() = default;
    Pic(const Pic& rhs) = default;
    Pic& operator=(const Pic& rhs) = default;
    Pic(Pic&& rhs) : m_x(rhs.m_x), m_y(rhs.m_y) {}
    Pic& operator=(Pic&& rhs)
    {
        if (this != &rhs) {
            m_x = rhs.m_x;
            m_y = rhs.m_y;
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const Pic& p)
    {
        os << "(" << p.m_x << "," << p.m_y << ")";
        return os;
    }

    int m_x;
    int m_y;

    struct Pre
    {
        bool operator()(const Pic& r1, const Pic& r2) const
        {
            return r2.m_x < r1.m_x;
        }
    };
};

template <typename T>
void pout(const T& t)
{
    for (const auto& it : t) {
        std::cout << it << ",";
    }
    std::cout << std::endl;
}

int main()
{
    //MinMaxHeap<int, std::greater<int>> heap;
    //MinMaxHeap<int> heap;
    MinMaxHeap<Pic, Pic::Pre> heap;
    heap.pushHeap(1);
    heap.pushHeap(2);
    heap.pushHeap(7);
    heap.pushHeap(0);
    heap.pushHeap(0);
    heap.pushHeap(6);
    heap.pushHeap(7);
    heap.pushHeap(6);
    heap.pushHeap(5);
    heap.pushHeap(std::move(Pic(31)));

    MinMaxHeap<Pic, Pic::Pre> copy_heap = std::move(heap);
    copy_heap.sortHeap();
    pout(copy_heap);
    std::cout << "========\n";

    while (!heap.empty()) {
        pout(heap);
        heap.popHeap();
    }

    return 0;
}