#ifndef _MIN_MAX_HEAP_H_
#define _MIN_MAX_HEAP_H_

#include <algorithm>
#include <vector>

template <typename T, typename C = std::less<T>>
class MinMaxHeap
{
public:
    typedef std::vector<T>                      Container;
    typedef C                                   Comparer;
    typedef T                                   ValueType;
    typedef ValueType&                          Reference;
    typedef const ValueType&                    ConstReference;
    typedef typename Container::iterator        Iterator;
    typedef typename Container::const_iterator  ConstIterator;
    typedef typename Container::size_type       SizeType;

public:
    MinMaxHeap() = default;
    ~MinMaxHeap() = default;
    MinMaxHeap(const MinMaxHeap& rhs) = default;
    MinMaxHeap& operator=(const MinMaxHeap& rhs) = default;

    MinMaxHeap(MinMaxHeap&& rhs) = default;
    MinMaxHeap& operator=(MinMaxHeap&& rhs) = default;
    /*
    MinMaxHeap(MinMaxHeap&& rhs) : m_container(std::move(rhs.m_container)) {}
    MinMaxHeap& operator=(MinMaxHeap&& rhs)
    {
        if (this != &rhs) {
            std::move(m_container = rhs.m_container);
        }
        return *this;
    }
    */

    bool empty() const
    {
        m_container.empty();
    }

    SizeType size() const
    {
        return m_container.size();
    }

    Reference head()
    {
        return *begin();
    }

    ConstReference head() const
    {
        return *begin();
    }

    Iterator begin()
    {
        return m_container.begin();
    }

    Iterator end()
    {
        return m_container.end();
    }

    ConstIterator begin() const
    {
        return m_container.begin();
    }

    ConstIterator end() const
    {
        return m_container.end();
    }

    void pushHeap(ValueType v)
    {
        m_container.push_back(v);
        std::push_heap(m_container.begin(), m_container.end(), Comparer());
    }

    void popHeap()
    {
        std::pop_heap(m_container.begin(), m_container.end(), Comparer());
        m_container.pop_back();
    }

    void sortHeap()
    {
        std::sort_heap(m_container.begin(), m_container.end(), Comparer());
    }
private:
    Container m_container;
};

#endif
