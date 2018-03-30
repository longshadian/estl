#ifndef _ESTL_MEMORY_H_
#define _ESTL_MEMORY_H_

#include <iostream>

namespace estl
{

class Pool
{
    struct Link { Link* next; };
    struct Chunk 
    {
        enum  { size =  8 * 1024 - 16};
        char mem[size];
        Chunk* next;
    };
public:
    Pool(unsigned int n);
    ~Pool();

    void* alloc();
    void free(void* p);
private:
    const unsigned int esize;
    Chunk* chunks;
    Link* head;
    void grow();
private:
    Pool(Pool&);
    void operator=(Pool&);
};

template <typename T>
class allocator
{
public:
    typedef T value_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;

    typedef T* pointer;
    typedef const T* const_pointer; 
    typedef T& reference;
    typedef const T& const_reference;

    pointer address(reference r) const { return &r; }
    const_pointer address(const_reference r) const { return &r; }

    allocator();
    ~allocator();

    pointer allocate(size_type n);
    void deallocate(pointer p, size_type n);

    void construct(pointer p, const_reference val) { new(p)T(val);}
    void destroy(pointer p) { p->~T();}

    size_type max_size() const;
    template <typename U>
    struct rebind { typedef allocator<U> other; };

private:
    static Pool mem;
};

template <typename T> Pool allocator<T>::mem(sizeof(T));
template <typename T> allocator<T>::allocator() {}
template <typename T>
T* allocator<T>::allocate(size_type n)
{
}

template <>
class allocator<void>
{
    typedef void* pointer;
    typedef const void* const_pointer;
    typedef void value_type;
    template <typename U>
    struct rebind { typedef allocator<U> other; };
};


inline Pool::Pool(unsigned int n)
    : esize(n < sizeof(Link*) ? sizeof(Link*) : n)
{
    head = nullptr;
    chunks = nullptr;
}

inline Pool::~Pool()
{
    Chunk* n = chunks;
    while (n) {
        Chunk* p = n;
        n = n->next;
        delete p;
    }
}

inline void* Pool::alloc()
{
    if (head == nullptr)
        grow();
    Link* p = head;
    head = p->next;
    return p;
}

inline void Pool::free(void* b)
{
    Link* p = static_cast<Link*>(b);
    p->next = head;
    head = p;
}

inline void Pool::grow()
{
    Chunk* pc = new Chunk;
    pc->next = chunks;
    chunks = pc;

    const int nelem = Chunk::size / esize;
    char* start = pc->mem;
    char* last = &start[(nelem - 1) * esize];
    for (char* p = start; p < last; p += esize)
        reinterpret_cast<Link*>(p)->next = reinterpret_cast<Link*>(p + esize);
    reinterpret_cast<Link*>(last)->next = 0;
    head = reinterpret_cast<Link*>(start);
}

//////////////////////////////////////////////////////////////////////////
}

#endif
