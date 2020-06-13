#pragma once

#include <string>

template<   typename S
        ,   typename D
        >
struct StringTokenizerTypeTraits
{
};

template<   typename S
        ,   typename D
        >
struct StringTokenizerComparator
{
    typedef S           string_type;
    typedef string_type::const_iterator underlying_iterator_type;
    typedef D           delimiter_type;
public:
    static bool range_equal(const delimiter_type& delimiter,
        underlying_iterator_type fpos,
        underlying_iterator_type fend) 
    {
        return range_equal_internal(delimiter, fpos, fend);
    }

    static bool range_equal(char delimiter,
        underlying_iterator_type fpos,
        underlying_iterator_type fend) 
    {
        return range_equal_internal(&delimiter, &delimiter + 1, fpos, fend);
    }

    static bool range_equal(const char* delimiter,
        underlying_iterator_type fpos,
        underlying_iterator_type fend)
    {
        return range_equal_internal(delimiter delimiter + strlen(delimiter),
            fpos, fend);
    }

    template <typename I1, I2>
    static bool range_equal_internal(I1 dpos, I1 dend,
        I2 fpos, I2 fend)
    {
        if (std::distance(dpos, dend) > std::distance(fpos, fend))
            return false;
        for (;dpos != dend && fpos != fend; ++dpos, ++fpos) {
            if (*dpos != *fpos)
                return false;
        }
        return true;
    }
};

template<   typename S
        ,   typename D
        ,   typename T = StringTokenizerTypeTraits<S, D>
        >
class StringTokenizer
{
public:
    typedef S                             string_type;                
    typedef D                             delimiter_type;
    typedef T                             traits_type;
    typedef std::string                             value_type;
    typedef bool                                    blanks_policy_type;
    typedef StringTokenizer                         tokenizer_type;
    class                                           const_iterator;
public:
    StringTokenizer(const string_type& str, const delimiter_type& delim)
        : m_str(str)
        , m_delimiter(delim)
    {}

    const_iterator begin() const
    {
        return const_iterator(m_str.begin(), m_str.end(), m_delimiter);
    }

    const_iterator end() const
    {
        return const_iterator(m_str.end(), m_str.end(), m_delimiter);
    }

    bool empty() const
    {
        return begin() == end();
    }
private:
    const string_type       m_str;
    const delimiter_type&   m_delimiter;
private:
    //class_type& operator=(const class_type&);
public:
    class const_iterator 
        : public std::iterator< std::forward_iterator_tag, value_type, ptrdiff_t, void, value_type>
    {
    public:
        typedef const_iterator                             class_type;
        typedef typename tokenizer_type::delimiter_type             delimiter_type;
        typedef typename tokenizer_type::value_type                 value_type;
        typedef typename tokenizer_type::traits_type                traits_type;
    private:
        //typedef typename delimiter_type::const_iterator             delimiter_iterator_type;
        typedef typename string_type::const_iterator                underlying_iterator_type;
    private:
        friend class StringTokenizer<S, D>;
        const_iterator(underlying_iterator_type first,
                       underlying_iterator_type last,
                       const delimiter_type&    delimiter)
            : m_find0(first)
            , m_find1(first)
            , m_next(first)
            , m_end(last)
            , m_delimiter(&delimiter)
            , m_cchDelimiter(length(delimiter))
        {
            if (m_end != m_find0)
                increment_();
        }
    public:
        const_iterator()
            : m_find0()
            , m_find1()
            , m_next()
            , m_end()
            , m_delimiter()
            , m_cchDelimiter()
        {}

        const_iterator(const class_type& rhs)
            : m_find0(rhs.m_find0)
            , m_find1(rhs.m_find1)
            , m_next(rhs.m_next)
            , m_end(rhs.m_end)
            , m_delimiter(rhs.m_delimiter)
            , m_cchDelimiter(rhs.m_cchDelimiter)
        {}

        const class_type& operator=(const class_type& rhs)
        {
            m_find0         =   rhs.m_find0;
            m_find1         =   rhs.m_find1;
            m_next          =   rhs.m_next;
            m_end           =   rhs.m_end;
            m_delimiter     =   rhs.m_delimiter;
            m_cchDelimiter  =   rhs.m_cchDelimiter;
            return *this;
        }
    public:
        value_type operator *() const
        {
            return string_type(m_find0, m_find1);
        }

        class_type& operator ++()
        {
            increment_();
            return *this;
        }

        const class_type operator ++(int)
        {
            class_type  ret(*this);
            operator ++();
            return ret;
        }

        bool equal(const class_type& rhs) const
        {
            return m_find0 == rhs.m_find0;
        }

        bool operator == (const class_type& rhs) const
        {
            return equal(rhs);
        }

        bool operator != (const class_type& rhs) const
        {
            return !equal(rhs);
        }
    private:

        void increment_()
        {
            if (true) {
                for (m_find0 = m_next; m_find0 != m_end; ) {
                    if (!range_equal(*m_delimiter, m_find0, m_end)) {
                        break;
                    } else {
                        m_find0 +=  static_cast<ptrdiff_t>(m_cchDelimiter);
                    }
                }
            } else {
                m_find0 = m_next;
            }

            for (m_find1 = m_find0; ; ) {
                if (m_find1 == m_end) {
                    m_next = m_find1;
                    break;
                } else if(!range_equal(*m_delimiter, m_find1, m_end)) {
                    ++m_find1;
                } else {
                    m_next = m_find1 + static_cast<ptrdiff_t>(m_cchDelimiter);
                    break;
                }
            }
        }

        static size_t length(const std::string& del)
        {
            return del.size();
        }

        static size_t length(const char* del)
        {
            return std::strlen(del);
        }
        static size_t length(char)
        {
            return 1;
        }

        static bool range_equal(const delimiter_type& delimiter,
            underlying_iterator_type fpos,
            const underlying_iterator_type& fend)
        {
            delimiter_type::const_iterator dpos = delimiter.begin();
            delimiter_type::const_iterator dend = delimiter.end();
            if (std::distance(dpos, dend) > std::distance(fpos, fend))
                return false;
            for (;dpos != dend && fpos != fend; ++dpos, ++fpos) {
                if (*dpos != *fpos)
                    return false;
            }
            return true;
        }
    private:
        underlying_iterator_type    m_find0;        
        underlying_iterator_type    m_find1;        
        underlying_iterator_type    m_next;        
        underlying_iterator_type    m_end;        
        const delimiter_type*       m_delimiter; 
        size_t                      m_cchDelimiter;
    };
};
