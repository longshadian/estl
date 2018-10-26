#pragma 

#include <vector>

#include "Token.h"

// punctuation ids
enum TokenIdentifie {
    P_RSHIFT_ASSIGN = 0,
    P_LSHIFT_ASSIGN,
    P_PARMS,
    P_PRECOMPMERGE,
    P_LOGIC_AND,
    P_LOGIC_OR,
    P_LOGIC_GEQ,
    P_LOGIC_LEQ,
    P_LOGIC_EQ,
    P_LOGIC_UNEQ,

    P_MUL_ASSIGN,
    P_DIV_ASSIGN,
    P_MOD_ASSIGN,
    P_ADD_ASSIGN,
    P_SUB_ASSIGN,
    P_INC,
    P_DEC,

    P_BIN_AND_ASSIGN,
    P_BIN_OR_ASSIGN,
    P_BIN_XOR_ASSIGN,
    P_RSHIFT,
    P_LSHIFT,

    P_POINTERREF,
    P_CPP1,
    P_CPP2,
    P_MUL,
    P_DIV,
    P_MOD,
    P_ADD,
    P_SUB,
    P_ASSIGN,

    P_BIN_AND,
    P_BIN_OR,
    P_BIN_XOR,
    P_BIN_NOT,

    P_LOGIC_NOT,
    P_LOGIC_GREATER,
    P_LOGIC_LESS,

    P_REF,
    P_COMMA,
    P_SEMICOLON,
    P_COLON,
    P_QUESTIONMARK,

    P_PARENTHESESOPEN,
    P_PARENTHESESCLOSE,
    P_BRACEOPEN,
    P_BRACECLOSE,
    P_SQBRACKETOPEN,
    P_SQBRACKETCLOSE,
    P_BACKSLASH,

    P_PRECOMP,
    P_DOLLAR,
};

// punctuation
struct Punctuation
{
    const char* m_p;  // punctuation character(s)
	int         m_n;   // punctuation id
};

class idLexer 
{
public:
                    // constructor
					idLexer();
					idLexer( int flags );
					idLexer( const char *filename, int flags = 0, bool OSPath = false );
					idLexer( const char *ptr, int length, const char *name, int flags = 0 );
					// destructor
					~idLexer();


public:
	int				LoadFile(const char* filename);
	int				GetToken(idToken* token);
	int				SkipWhiteSpaceAndComment();

    int             ParseIdentifier(idToken* token);
    int             ParseNumber(idToken* token);


private:
    std::vector<char>   m_buffer;
    const char*         m_current_p;    // current pointer in the script
    int                 m_line;

public:

					// load a script from the given file at the given offset with the given length
	int				LoadFile( const char *filename, bool OSPath = false );
					// load a script from the given memory with the given length and a specified line offset,
					// so source strings extracted from a file can still refer to proper line numbers in the file
					// NOTE: the ptr is expected to point at a valid C string: ptr[length] == '\0'
	int				LoadMemory( const char *ptr, int length, const char *name, int startLine = 1 );
					// free the script
	void			FreeSource( void );
					// returns true if a script is loaded
	int				IsLoaded( void ) { return idLexer::loaded; };
					// read a token
	int				ReadToken( idToken *token );
					// expect a certain token, reads the token when available
	int				ExpectTokenString( const char *string );
					// expect a certain token type
	int				ExpectTokenType( int type, int subtype, idToken *token );
					// expect a token
	int				ExpectAnyToken( idToken *token );
					// returns true when the token is available
	int				CheckTokenString( const char *string );
					// returns true an reads the token when a token with the given type is available
	int				CheckTokenType( int type, int subtype, idToken *token );
					// returns true if the next token equals the given string but does not remove the token from the source
	int				PeekTokenString( const char *string );
					// returns true if the next token equals the given type but does not remove the token from the source
	int				PeekTokenType( int type, int subtype, idToken *token );
					// skip tokens until the given token string is read
	int				SkipUntilString( const char *string );
					// skip the rest of the current line
	int				SkipRestOfLine( void );
					// skip the braced section
	int				SkipBracedSection( bool parseFirstBrace = true );
					// unread the given token
	void			UnreadToken( const idToken *token );
					// read a token only if on the same line
	int				ReadTokenOnLine( idToken *token );
		
					//Returns the rest of the current line
	const char*		ReadRestOfLine(idStr& out);

					// read a signed integer
	int				ParseInt( void );
					// read a boolean
	bool			ParseBool( void );
					// read a floating point number.  If errorFlag is NULL, a non-numeric token will
					// issue an Error().  If it isn't NULL, it will issue a Warning() and set *errorFlag = true
	float			ParseFloat( bool *errorFlag = NULL );
					// parse matrices with floats
	int				Parse1DMatrix( int x, float *m );
	int				Parse2DMatrix( int y, int x, float *m );
	int				Parse3DMatrix( int z, int y, int x, float *m );
					// parse a braced section into a string
	const char *	ParseBracedSection( idStr &out );
					// parse a braced section into a string, maintaining indents and newlines
	const char *	ParseBracedSectionExact ( idStr &out, int tabs = -1 );
					// parse the rest of the line
	const char *	ParseRestOfLine( idStr &out );
					// retrieves the white space characters before the last read token
	int				GetLastWhiteSpace( idStr &whiteSpace ) const;
					// returns start index into text buffer of last white space
	int				GetLastWhiteSpaceStart( void ) const;
					// returns end index into text buffer of last white space
	int				GetLastWhiteSpaceEnd( void ) const;
					// set an array with punctuations, NULL restores default C/C++ set, see default_punctuations for an example
	void			SetPunctuations( const punctuation_t *p );
					// returns a pointer to the punctuation with the given id
	const char *	GetPunctuationFromId( int id );
					// get the id for the given punctuation
	int				GetPunctuationId( const char *p );
					// set lexer flags
	void			SetFlags( int flags );
					// get lexer flags
	int				GetFlags( void );
					// reset the lexer
	void			Reset( void );
					// returns true if at the end of the file
	int				EndOfFile( void );
					// returns the current filename
	const char *	GetFileName( void );
					// get offset in script
	const int		GetFileOffset( void );
					// get file time
	const ID_TIME_T	GetFileTime( void );
					// returns the current line number
	const int		GetLineNum( void );
					// print an error message
	void			Error( const char *str, ... ) id_attribute((format(printf,2,3)));
					// print a warning message
	void			Warning( const char *str, ... ) id_attribute((format(printf,2,3)));
					// returns true if Error() was called with LEXFL_NOFATALERRORS or LEXFL_NOERRORS set
	bool			HadError( void ) const;

					// set the base folder to load files from
	static void		SetBaseFolder( const char *path );

private:
	int				loaded;					// set when a script file is loaded from file or memory
	idStr			filename;				// file name of the script
	int				allocated;				// true if buffer memory was allocated
	const char *	buffer;					// buffer containing the script
	const char *	script_p;				// current pointer in the script
	const char *	end_p;					// pointer to the end of the script
	const char *	lastScript_p;			// script pointer before reading token
	const char *	whiteSpaceStart_p;		// start of last white space
	const char *	whiteSpaceEnd_p;		// end of last white space
	ID_TIME_T			fileTime;				// file time
	int				length;					// length of the script in bytes
	int				line;					// current line in script
	int				lastline;				// line before reading token
	int				tokenavailable;			// set by unreadToken
	int				flags;					// several script flags
	const punctuation_t *punctuations;		// the punctuations used in the script
	int *			punctuationtable;		// ASCII table with punctuations
	int *			nextpunctuation;		// next punctuation in chain
	idToken			token;					// available token
	idLexer *		next;					// next script in a chain
	bool			hadError;				// set by idLexer::Error, even if the error is supressed

	static char		baseFolder[ 256 ];		// base folder to load files from

private:
	void			CreatePunctuationTable( const punctuation_t *punctuations );
	int				ReadWhiteSpace( void );
	int				ReadEscapeCharacter( char *ch );
	int				ReadString( idToken *token, int quote );
	int				ReadName( idToken *token );
	int				ReadNumber( idToken *token );
	int				ReadPunctuation( idToken *token );
	int				ReadPrimitive( idToken *token );
	int				CheckString( const char *str ) const;
	int				NumLinesCrossed( void );
};

ID_INLINE const char *idLexer::GetFileName( void ) {
	return idLexer::filename;
}

ID_INLINE const int idLexer::GetFileOffset( void ) {
	return idLexer::script_p - idLexer::buffer;
}

ID_INLINE const ID_TIME_T idLexer::GetFileTime( void ) {
	return idLexer::fileTime;
}

ID_INLINE const int idLexer::GetLineNum( void ) {
	return idLexer::line;
}

ID_INLINE void idLexer::SetFlags( int flags ) {
	idLexer::flags = flags;
}

ID_INLINE int idLexer::GetFlags( void ) {
	return idLexer::flags;
}
