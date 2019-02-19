#include "Utils.h"

#include <cstring>
#include <algorithm>

namespace mysqlcpp {

namespace util {

Tokenizer::Tokenizer(const std::string &src, const char sep)
    : m_str()
    , m_storage()
{
    m_str.resize(src.length() + 1);
    std::copy(src.begin(), src.end(), m_str.begin());

    char* pos_old = m_str.data();
    char* pos = m_str.data();
    for (;;) {
        if (*pos == sep) {
            if (pos_old != pos)
                m_storage.push_back(pos_old);

            pos_old = pos + 1;
            *pos = '\0';
        } else if (*pos == '\0') {
            if (pos_old != pos)
                m_storage.push_back(pos_old);
            break;
        }
        ++pos;
    }
}

}
}


