#include "Token.h"

#include "Math.h"


const std::string g_token_type_string[] = 
{
    "unknown",
    "string",
    "literal",
    "number",
    "identifier",
    "punctuation",
    "keyword",
    "eof",
};

idToken::idToken()
    : m_type(TokenType::Unknown)
    , m_subtype()
    , m_line()
    , m_linesCrossed()
    , m_flags()
    , m_buffer()

    , m_intvalue()
    , m_floatvalue()
    , m_whiteSpaceStart_p()
    , m_whiteSpaceEnd_p()
    , m_next()
{
}

idToken::~idToken()
{
}

void idToken::Reset()
{
    m_type = TokenType::Unknown;
    m_subtype = 0;
    m_line = 0;
    m_linesCrossed = 0;
    m_flags = 0;
    m_buffer.clear();

    m_intvalue = 0;
    m_floatvalue = 0;
    m_whiteSpaceStart_p = nullptr;
    m_whiteSpaceEnd_p = nullptr;
    m_next = nullptr;
} 

void idToken::AppendCharacter(char a)
{
    m_buffer.push_back(a);
}

int idToken::Length() const
{
    return static_cast<int>(m_buffer.size());
}

char idToken::GetCharacter(size_t pos) const
{
    return m_buffer[pos];
}

std::string_view idToken::AsStringView() const
{
    if (m_buffer.empty())
        return std::string_view();
    return std::string_view(m_buffer.data(), m_buffer.size());
}

std::string_view idToken::AsTypeStringView() const
{
    return g_token_type_string[ static_cast<std::underlying_type<TokenType>::type>(m_type)];
}

/*
================
idToken::NumberValue
================
*/
void idToken::NumberValue()
{
	int i, pow, div, c;
	const char *p;
	double m;

	assert(m_type == TokenType::Number);
    // TODO
	//p = c_str();
	m_floatvalue = 0;
	m_intvalue = 0;
	// floating point number
	if ( m_subtype & TT_FLOAT ) {
		if ( m_subtype & ( TT_INFINITE | TT_INDEFINITE | TT_NAN ) ) {
			if ( m_subtype & TT_INFINITE ) {			// 1.#INF
				unsigned int inf = 0x7f800000;
				m_floatvalue = (double) *(float*)&inf;
			}
			else if ( m_subtype & TT_INDEFINITE ) {	// 1.#IND
				unsigned int ind = 0xffc00000;
				m_floatvalue = (double) *(float*)&ind;
			}
			else if ( m_subtype & TT_NAN ) {			// 1.#QNAN
				unsigned int nan = 0x7fc00000;
				m_floatvalue = (double) *(float*)&nan;
			}
		}
		else {
			while( *p && *p != '.' && *p != 'e' ) {
				m_floatvalue = m_floatvalue * 10.0 + (double) (*p - '0');
				p++;
			}
			if ( *p == '.' ) {
				p++;
				for( m = 0.1; *p && *p != 'e'; p++ ) {
					m_floatvalue = m_floatvalue + (double) (*p - '0') * m;
					m *= 0.1;
				}
			}
			if ( *p == 'e' ) {
				p++;
				if ( *p == '-' ) {
					div = true;
					p++;
				}
				else if ( *p == '+' ) {
					div = false;
					p++;
				}
				else {
					div = false;
				}
				pow = 0;
				for ( pow = 0; *p; p++ ) {
					pow = pow * 10 + (int) (*p - '0');
				}
				for ( m = 1.0, i = 0; i < pow; i++ ) {
					m *= 10.0;
				}
				if ( div ) {
					m_floatvalue /= m;
				}
				else {
					m_floatvalue *= m;
				}
			}
		}
		m_intvalue = idMath::Ftol( m_floatvalue );
	}
	else if ( m_subtype & TT_DECIMAL ) {
		while( *p ) {
			m_intvalue = m_intvalue * 10 + (*p - '0');
			p++;
		}
		m_floatvalue = m_intvalue;
	}
	else if ( m_subtype & TT_IPADDRESS ) {
		c = 0;
		while( *p && *p != ':' ) {
			if ( *p == '.' ) {
				while( c != 3 ) {
					m_intvalue = m_intvalue * 10;
					c++;
				}
				c = 0;
			}
			else {
				m_intvalue = m_intvalue * 10 + (*p - '0');
				c++;
			}
			p++;
		}
		while( c != 3 ) {
			m_intvalue = m_intvalue * 10;
			c++;
		}
		m_floatvalue = m_intvalue;
	}
	else if ( m_subtype & TT_OCTAL ) {
		// step over the first zero
		p += 1;
		while( *p ) {
			m_intvalue = (m_intvalue << 3) + (*p - '0');
			p++;
		}
		m_floatvalue = m_intvalue;
	}
	else if ( m_subtype & TT_HEX ) {
		// step over the leading 0x or 0X
		p += 2;
		while( *p ) {
			m_intvalue <<= 4;
			if (*p >= 'a' && *p <= 'f')
				m_intvalue += *p - 'a' + 10;
			else if (*p >= 'A' && *p <= 'F')
				m_intvalue += *p - 'A' + 10;
			else
				m_intvalue += *p - '0';
			p++;
		}
		m_floatvalue = m_intvalue;
	}
	else if ( m_subtype & TT_BINARY ) {
		// step over the leading 0b or 0B
		p += 2;
		while( *p ) {
			m_intvalue = (m_intvalue << 1) + (*p - '0');
			p++;
		}
		m_floatvalue = m_intvalue;
	}
	m_subtype |= TT_VALUESVALID;
}

/*
================
idToken::ClearTokenWhiteSpace
================
*/
void idToken::ClearTokenWhiteSpace()
{
	m_whiteSpaceStart_p = NULL;
	m_whiteSpaceEnd_p = NULL;
	m_linesCrossed = 0;
}
