#pragma once

#include <string>
#include <vector>

enum class TokenType
{
    Unknown =           0,      // 词法分析器无法分析，暂且认为是合法的c++ token类型。
    String =			1,		// string    start with "
    Literal =		    2,		// literal   start with '
    Number =			3,		// number
    Identifier =		4,		// name
    Puncatuation =	    5,		// punctuation
    Keyword =           6,      // keyword
    Eof =               7,      // end of file
};

enum TokenSubType
{
    TT_INTEGER =					0x00001,		// integer
    TT_DECIMAL =					0x00002,		// decimal number
    TT_HEX =						0x00004,		// hexadecimal number
    TT_OCTAL =  					0x00008,		// octal number
    TT_BINARY = 					0x00010,		// binary number
    TT_LONG = 						0x00020,		// long int
    TT_UNSIGNED =					0x00040,		// unsigned int
    TT_FLOAT =  					0x00080,		// floating point number
    TT_SINGLE_PRECISION = 			0x00100,		// float
    TT_DOUBLE_PRECISION = 			0x00200,		// double
    TT_EXTENDED_PRECISION = 		0x00400,		// long double
    TT_INFINITE =					0x00800,		// infinite 1.#INF
    TT_INDEFINITE = 				0x01000,		// indefinite 1.#IND
    TT_NAN = 						0x02000,		// NaN
    TT_IPADDRESS =  				0x04000,		// ip address
    TT_IPPORT = 					0x08000,		// ip port
    TT_VALUESVALID = 				0x10000,		// set if intvalue and floatvalue are valid
};

// string sub type is the length of the string
// literal sub type is the ASCII code
// punctuation sub type is the punctuation id
// identifier sub type is the length of the name

class idLexer;

class idToken 
{
    friend class idLexer;

public:
    TokenType       m_type;								// token type
	int				m_subtype;							// token sub type
	int				m_line;								// line in script the token was on
	int				m_linesCrossed;						// number of lines crossed in white space before token
	int				m_flags;						    // token flags, used for recursive defines
    std::vector<char>   m_buffer;

					idToken();
					~idToken();
					idToken(const idToken&) = delete;
					idToken& operator=(const idToken&) = delete;
					idToken(idToken&&) = delete;
					idToken& operator=(idToken&&) = delete;
    void            Reset();
    void			AppendCharacter(char a);
    int             Length() const;
    char            GetCharacter(size_t pos) const;
    std::string_view AsStringView() const;
    std::string_view AsTypeStringView() const;

	double			GetDoubleValue();				// double value of TT_NUMBER
	float			GetFloatValue();				// float value of TT_NUMBER
	unsigned long	GetUnsignedLongValue();		// unsigned long value of TT_NUMBER
	int				GetIntValue();				// int value of TT_NUMBER
	int				WhiteSpaceBeforeToken() const;// returns length of whitespace before token
	void			ClearTokenWhiteSpace();		// forget whitespace before token

	void			NumberValue();				// calculate values for a TT_NUMBER

private:
	unsigned long	m_intvalue;							// integer value
	double			m_floatvalue;                       // floating point value
	const char *	m_whiteSpaceStart_p;                // start of white space before token, only used by idLexer
	const char *	m_whiteSpaceEnd_p;					// end of white space before token, only used by idLexer
	idToken *		m_next;								// next token in chain, only used by idParser
};

inline double idToken::GetDoubleValue() 
{
	if (m_type != TokenType::Number) {
		return 0.0;
	}
	if ( !(m_subtype & TT_VALUESVALID) ) {
		NumberValue();
	}
	return m_floatvalue;
}

inline float idToken::GetFloatValue()
{
	return (float) GetDoubleValue();
}

inline unsigned long idToken::GetUnsignedLongValue()
{
	if (m_type != TokenType::Number) {
		return 0;
	}
	if ( !(m_subtype & TT_VALUESVALID) ) {
		NumberValue();
	}
	return m_intvalue;
}

inline int idToken::GetIntValue()
{
	return (int) GetUnsignedLongValue();
}

inline int idToken::WhiteSpaceBeforeToken() const
{
	return ( m_whiteSpaceEnd_p > m_whiteSpaceStart_p );
}
