#ifndef _LIBCPP_VECTOR
#define _LIBCPP_VECTOR

/*
    vector synopsis

namespace std
{

template <class T, class Allocator = allocator<T> >
class vector
{
public:
    typedef T                                        value_type;
    typedef Allocator                                allocator_type;
    typedef typename allocator_type::reference       reference;
    typedef typename allocator_type::const_reference const_reference;
    typedef implementation-defined                   iterator;
    typedef implementation-defined                   const_iterator;
    typedef typename allocator_type::size_type       size_type;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::pointer         pointer;
    typedef typename allocator_type::const_pointer   const_pointer;
    typedef std::reverse_iterator<iterator>          reverse_iterator;
    typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;

    vector()
        noexcept(is_nothrow_default_constructible<allocator_type>::value);
    explicit vector(const allocator_type&);
    explicit vector(size_type n);
    explicit vector(size_type n, const allocator_type&); // C++14
    vector(size_type n, const value_type& value, const allocator_type& = allocator_type());
    template <class InputIterator>
        vector(InputIterator first, InputIterator last, const allocator_type& = allocator_type());
    vector(const vector& x);
    vector(vector&& x)
        noexcept(is_nothrow_move_constructible<allocator_type>::value);
    vector(initializer_list<value_type> il);
    vector(initializer_list<value_type> il, const allocator_type& a);
    ~vector();
    vector& operator=(const vector& x);
    vector& operator=(vector&& x)
        noexcept(
             allocator_type::propagate_on_container_move_assignment::value &&
             is_nothrow_move_assignable<allocator_type>::value);
    vector& operator=(initializer_list<value_type> il);
    template <class InputIterator>
        void assign(InputIterator first, InputIterator last);
    void assign(size_type n, const value_type& u);
    void assign(initializer_list<value_type> il);

    allocator_type get_allocator() const noexcept;

    iterator               begin() noexcept;
    const_iterator         begin()   const noexcept;
    iterator               end() noexcept;
    const_iterator         end()     const noexcept;

    reverse_iterator       rbegin() noexcept;
    const_reverse_iterator rbegin()  const noexcept;
    reverse_iterator       rend() noexcept;
    const_reverse_iterator rend()    const noexcept;

    const_iterator         cbegin()  const noexcept;
    const_iterator         cend()    const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend()   const noexcept;

    size_type size() const noexcept;
    size_type max_size() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;
    void reserve(size_type n);
    void shrink_to_fit() noexcept;

    reference       operator[](size_type n);
    const_reference operator[](size_type n) const;
    reference       at(size_type n);
    const_reference at(size_type n) const;

    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;

    value_type*       data() noexcept;
    const value_type* data() const noexcept;

    void push_back(const value_type& x);
    void push_back(value_type&& x);
    template <class... Args>
        void emplace_back(Args&&... args);
    void pop_back();

    template <class... Args> iterator emplace(const_iterator position, Args&&... args);
    iterator insert(const_iterator position, const value_type& x);
    iterator insert(const_iterator position, value_type&& x);
    iterator insert(const_iterator position, size_type n, const value_type& x);
    template <class InputIterator>
        iterator insert(const_iterator position, InputIterator first, InputIterator last);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

    void clear() noexcept;

    void resize(size_type sz);
    void resize(size_type sz, const value_type& c);

    void swap(vector&)
        noexcept(!allocator_type::propagate_on_container_swap::value ||
                 __is_nothrow_swappable<allocator_type>::value);

    bool __invariants() const;
};

template <class Allocator = allocator<T> >
class vector<bool, Allocator>
{
public:
    typedef bool                                     value_type;
    typedef Allocator                                allocator_type;
    typedef implementation-defined                   iterator;
    typedef implementation-defined                   const_iterator;
    typedef typename allocator_type::size_type       size_type;
    typedef typename allocator_type::difference_type difference_type;
    typedef iterator                                 pointer;
    typedef const_iterator                           const_pointer;
    typedef std::reverse_iterator<iterator>          reverse_iterator;
    typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;

    class reference
    {
    public:
        reference(const reference&) noexcept;
        operator bool() const noexcept;
        reference& operator=(const bool x) noexcept;
        reference& operator=(const reference& x) noexcept;
        iterator operator&() const noexcept;
        void flip() noexcept;
    };

    class const_reference
    {
    public:
        const_reference(const reference&) noexcept;
        operator bool() const noexcept;
        const_iterator operator&() const noexcept;
    };

    vector()
        noexcept(is_nothrow_default_constructible<allocator_type>::value);
    explicit vector(const allocator_type&);
    explicit vector(size_type n, const allocator_type& a = allocator_type()); // C++14
    vector(size_type n, const value_type& value, const allocator_type& = allocator_type());
    template <class InputIterator>
        vector(InputIterator first, InputIterator last, const allocator_type& = allocator_type());
    vector(const vector& x);
    vector(vector&& x)
        noexcept(is_nothrow_move_constructible<allocator_type>::value);
    vector(initializer_list<value_type> il);
    vector(initializer_list<value_type> il, const allocator_type& a);
    ~vector();
    vector& operator=(const vector& x);
    vector& operator=(vector&& x)
        noexcept(
             allocator_type::propagate_on_container_move_assignment::value &&
             is_nothrow_move_assignable<allocator_type>::value);
    vector& operator=(initializer_list<value_type> il);
    template <class InputIterator>
        void assign(InputIterator first, InputIterator last);
    void assign(size_type n, const value_type& u);
    void assign(initializer_list<value_type> il);

    allocator_type get_allocator() const noexcept;

    iterator               begin() noexcept;
    const_iterator         begin()   const noexcept;
    iterator               end() noexcept;
    const_iterator         end()     const noexcept;

    reverse_iterator       rbegin() noexcept;
    const_reverse_iterator rbegin()  const noexcept;
    reverse_iterator       rend() noexcept;
    const_reverse_iterator rend()    const noexcept;

    const_iterator         cbegin()  const noexcept;
    const_iterator         cend()    const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend()   const noexcept;

    size_type size() const noexcept;
    size_type max_size() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;
    void reserve(size_type n);
    void shrink_to_fit() noexcept;

    reference       operator[](size_type n);
    const_reference operator[](size_type n) const;
    reference       at(size_type n);
    const_reference at(size_type n) const;

    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;

    void push_back(const value_type& x);
    template <class... Args> void emplace_back(Args&&... args);  // C++14
    void pop_back();

    template <class... Args> iterator emplace(const_iterator position, Args&&... args);  // C++14
    iterator insert(const_iterator position, const value_type& x);
    iterator insert(const_iterator position, size_type n, const value_type& x);
    template <class InputIterator>
        iterator insert(const_iterator position, InputIterator first, InputIterator last);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

    void clear() noexcept;

    void resize(size_type sz);
    void resize(size_type sz, value_type x);

    void swap(vector&)
        noexcept(!allocator_type::propagate_on_container_swap::value ||
                 __is_nothrow_swappable<allocator_type>::value);
    void flip() noexcept;

    bool __invariants() const;
};

template <class Allocator> struct hash<std::vector<bool, Allocator>>;

template <class T, class Allocator> bool operator==(const vector<T,Allocator>& x, const vector<T,Allocator>& y);
template <class T, class Allocator> bool operator< (const vector<T,Allocator>& x, const vector<T,Allocator>& y);
template <class T, class Allocator> bool operator!=(const vector<T,Allocator>& x, const vector<T,Allocator>& y);
template <class T, class Allocator> bool operator> (const vector<T,Allocator>& x, const vector<T,Allocator>& y);
template <class T, class Allocator> bool operator>=(const vector<T,Allocator>& x, const vector<T,Allocator>& y);
template <class T, class Allocator> bool operator<=(const vector<T,Allocator>& x, const vector<T,Allocator>& y);

template <class T, class Allocator>
void swap(vector<T,Allocator>& x, vector<T,Allocator>& y)
    noexcept(noexcept(x.swap(y)));

}  // std

*/

//**************************************************************************************************
namespace estl
{

template <class T, class Allocator = allocator<T> >
class vector
{
public:
    typedef T                                        value_type;
    typedef Allocator                                allocator_type;
    typedef typename allocator_type::reference       reference;
    typedef typename allocator_type::const_reference const_reference;
    typedef implementation-defined                   iterator;
    typedef implementation-defined                   const_iterator;
    typedef typename allocator_type::size_type       size_type;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::pointer         pointer;
    typedef typename allocator_type::const_pointer   const_pointer;
    typedef std::reverse_iterator<iterator>          reverse_iterator;
    typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;

    vector()
        noexcept(is_nothrow_default_constructible<allocator_type>::value);
    explicit vector(const allocator_type&);
    explicit vector(size_type n);
    explicit vector(size_type n, const allocator_type&); // C++14
    vector(size_type n, const value_type& value, const allocator_type& = allocator_type());
    template <class InputIterator>
        vector(InputIterator first, InputIterator last, const allocator_type& = allocator_type());
    vector(const vector& x);
    vector(vector&& x)
        noexcept(is_nothrow_move_constructible<allocator_type>::value);
    vector(initializer_list<value_type> il);
    vector(initializer_list<value_type> il, const allocator_type& a);
    ~vector();
    vector& operator=(const vector& x);
    vector& operator=(vector&& x)
        noexcept(
             allocator_type::propagate_on_container_move_assignment::value &&
             is_nothrow_move_assignable<allocator_type>::value);
    vector& operator=(initializer_list<value_type> il);
    template <class InputIterator>
        void assign(InputIterator first, InputIterator last);
    void assign(size_type n, const value_type& u);
    void assign(initializer_list<value_type> il);

    allocator_type get_allocator() const noexcept;

    iterator               begin() noexcept;
    const_iterator         begin()   const noexcept;
    iterator               end() noexcept;
    const_iterator         end()     const noexcept;

    reverse_iterator       rbegin() noexcept;
    const_reverse_iterator rbegin()  const noexcept;
    reverse_iterator       rend() noexcept;
    const_reverse_iterator rend()    const noexcept;

    const_iterator         cbegin()  const noexcept;
    const_iterator         cend()    const noexcept;
    const_reverse_iterator crbegin() const noexcept;
    const_reverse_iterator crend()   const noexcept;

    size_type size() const noexcept;
    size_type max_size() const noexcept;
    size_type capacity() const noexcept;
    bool empty() const noexcept;
    void reserve(size_type n);
    void shrink_to_fit() noexcept;

    reference       operator[](size_type n);
    const_reference operator[](size_type n) const;
    reference       at(size_type n);
    const_reference at(size_type n) const;

    reference       front();
    const_reference front() const;
    reference       back();
    const_reference back() const;

    value_type*       data() noexcept;
    const value_type* data() const noexcept;

    void push_back(const value_type& x);
    void push_back(value_type&& x);
    template <class... Args>
        void emplace_back(Args&&... args);
    void pop_back();

    template <class... Args> iterator emplace(const_iterator position, Args&&... args);
    iterator insert(const_iterator position, const value_type& x);
    iterator insert(const_iterator position, value_type&& x);
    iterator insert(const_iterator position, size_type n, const value_type& x);
    template <class InputIterator>
        iterator insert(const_iterator position, InputIterator first, InputIterator last);
    iterator insert(const_iterator position, initializer_list<value_type> il);

    iterator erase(const_iterator position);
    iterator erase(const_iterator first, const_iterator last);

    void clear() noexcept;

    void resize(size_type sz);
    void resize(size_type sz, const value_type& c);

    void swap(vector&)
        noexcept(!allocator_type::propagate_on_container_swap::value ||
                 __is_nothrow_swappable<allocator_type>::value);

    bool __invariants() const;
private:
	size_type newSize();
private:
	allocator<T>	mAllocator;
	iterator		mBegin;
	iterator		mEnd;
	size_type		mSize;
};

template <typename T, typename Alloc>
typename vector<T, Alloc>::size_type vector<T, Alloc>::newSize()
{
	return mSize << 1;
}

template <typename T, typename Alloc>
void vector<T, Alloc>::push_back(const value_type& x)
{
	if (mSize == 0) {
		size_type new_size = newSize();
		iterator = mAllocator.allocate();
		mBegin = mAllocator.allocate();
	}

	iterator p = mBegin;
	if (mEnd - mBegin == 0) {
		mBegin = mAllocator.allocate();

	}
}

//////////////////////////////////////////////////////////////////////////
}

#endif  // _LIBCPP_VECTOR
