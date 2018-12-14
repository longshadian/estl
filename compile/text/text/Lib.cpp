#include "Lib.h"

#include "Math.h"

#include <cassert>

/*
#if defined( MACOS_X )
#include <signal.h>
#include <sys/types.h>
#include <unistd.h>
#endif
*/

/*
================
idLib::Init
================
*/
void idLib::Init()
{
	assert( sizeof( bool ) == 1 );

	// initialize little/big endian conversion
	Swap_Init();

	// initialize math
	idMath::Init();
}

/*
================
idLib::ShutDown
================
*/
void idLib::ShutDown()
{
}

/*
===============================================================================

	Colors

===============================================================================
*/

static dword colorMask[2] = { 255, 0 };

/*
================
ColorFloatToByte
================
*/
static 
byte ColorFloatToByte(float c)
{
	return (byte) ( ( (dword) ( c * 255.0f ) ) & colorMask[FLOATSIGNBITSET(c)] );
}

/*
===============================================================================

	Byte order functions

===============================================================================
*/

// can't just use function pointers, or dll linkage can mess up
static short	(*_BigShort)( short l );
static short	(*_LittleShort)( short l );
static int		(*_BigLong)( int l );
static int		(*_LittleLong)( int l );
static float	(*_BigFloat)( float l );
static float	(*_LittleFloat)( float l );
static void		(*_BigRevBytes)( void *bp, int elsize, int elcount );
static void		(*_LittleRevBytes)( void *bp, int elsize, int elcount );
static void     (*_LittleBitField)( void *bp, int elsize );
static void		(*_SixtetsForInt)( byte *out, int src );
static int		(*_IntForSixtets)( byte *in );

short	BigShort( short l ) { return _BigShort( l ); }
short	LittleShort( short l ) { return _LittleShort( l ); }
int		BigLong( int l ) { return _BigLong( l ); }
int		LittleLong( int l ) { return _LittleLong( l ); }
float	BigFloat( float l ) { return _BigFloat( l ); }
float	LittleFloat( float l ) { return _LittleFloat( l ); }
void	BigRevBytes( void *bp, int elsize, int elcount ) { _BigRevBytes( bp, elsize, elcount ); }
void	LittleRevBytes( void *bp, int elsize, int elcount ){ _LittleRevBytes( bp, elsize, elcount ); }
void	LittleBitField( void *bp, int elsize ){ _LittleBitField( bp, elsize ); }

void	SixtetsForInt( byte *out, int src) { _SixtetsForInt( out, src ); }
int		IntForSixtets( byte *in ) { return _IntForSixtets( in ); }

/*
================
ShortSwap
================
*/
short ShortSwap( short l ) {
	byte    b1,b2;

	b1 = l&255;
	b2 = (l>>8)&255;

	return (b1<<8) + b2;
}

/*
================
ShortNoSwap
================
*/
short ShortNoSwap( short l ) {
	return l;
}

/*
================
LongSwap
================
*/
int LongSwap ( int l ) {
	byte    b1,b2,b3,b4;

	b1 = l&255;
	b2 = (l>>8)&255;
	b3 = (l>>16)&255;
	b4 = (l>>24)&255;

	return ((int)b1<<24) + ((int)b2<<16) + ((int)b3<<8) + b4;
}

/*
================
LongNoSwap
================
*/
int	LongNoSwap( int l ) {
	return l;
}

/*
================
FloatSwap
================
*/
float FloatSwap( float f ) {
	union {
		float	f;
		byte	b[4];
	} dat1, dat2;
	
	
	dat1.f = f;
	dat2.b[0] = dat1.b[3];
	dat2.b[1] = dat1.b[2];
	dat2.b[2] = dat1.b[1];
	dat2.b[3] = dat1.b[0];
	return dat2.f;
}

/*
================
FloatNoSwap
================
*/
float FloatNoSwap( float f ) {
	return f;
}

/*
=====================================================================
RevBytesSwap

Reverses byte order in place.

INPUTS
   bp       bytes to reverse
   elsize   size of the underlying data type
   elcount  number of elements to swap

RESULTS
   Reverses the byte order in each of elcount elements.
===================================================================== */
void RevBytesSwap( void *bp, int elsize, int elcount ) {
	unsigned char *p, *q;

	p = ( unsigned char * ) bp;

	if ( elsize == 2 ) {
		q = p + 1;
		while ( elcount-- ) {
			*p ^= *q;
			*q ^= *p;
			*p ^= *q;
			p += 2;
			q += 2;
		}
		return;
	}

	while ( elcount-- ) {
		q = p + elsize - 1;
		while ( p < q ) {
			*p ^= *q;
			*q ^= *p;
			*p ^= *q;
			++p;
			--q;
		}
		p += elsize >> 1;
	}
}

/*
 =====================================================================
 RevBytesSwap
 
 Reverses byte order in place, then reverses bits in those bytes
 
 INPUTS
 bp       bitfield structure to reverse
 elsize   size of the underlying data type
 
 RESULTS
 Reverses the bitfield of size elsize.
 ===================================================================== */
void RevBitFieldSwap( void *bp, int elsize) {
	int i;
	unsigned char *p, t, v;
	
	LittleRevBytes( bp, elsize, 1 );
	
	p = (unsigned char *) bp;
	while ( elsize-- ) {
		v = *p;
		t = 0;
		for (i = 7; i; i--) {
			t <<= 1;
			v >>= 1;
			t |= v & 1;
		}
		*p++ = t;
	}
}

/*
================
RevBytesNoSwap
================
*/
void RevBytesNoSwap( void *bp, int elsize, int elcount ) {
	return;
}

/*
 ================
 RevBytesNoSwap
 ================
 */
void RevBitFieldNoSwap( void *bp, int elsize ) {
	return;
}

/*
================
SixtetsForIntLittle
================
*/
void SixtetsForIntLittle( byte *out, int src) {
	byte *b = (byte *)&src;
	out[0] = ( b[0] & 0xfc ) >> 2;
	out[1] = ( ( b[0] & 0x3 ) << 4 ) + ( ( b[1] & 0xf0 ) >> 4 );
	out[2] = ( ( b[1] & 0xf ) << 2 ) + ( ( b[2] & 0xc0 ) >> 6 );
	out[3] = b[2] & 0x3f;
}

/*
================
SixtetsForIntBig
TTimo: untested - that's the version from initial base64 encode
================
*/
void SixtetsForIntBig( byte *out, int src) {
	for( int i = 0 ; i < 4 ; i++ ) {
		out[i] = src & 0x3f;
		src >>= 6;
	}
}

/*
================
IntForSixtetsLittle
================
*/
int IntForSixtetsLittle( byte *in ) {
	int ret = 0;
	byte *b = (byte *)&ret;
	b[0] |= in[0] << 2;
	b[0] |= ( in[1] & 0x30 ) >> 4;
	b[1] |= ( in[1] & 0xf ) << 4;
	b[1] |= ( in[2] & 0x3c ) >> 2;
	b[2] |= ( in[2] & 0x3 ) << 6;
	b[2] |= in[3];
	return ret;
}

/*
================
IntForSixtetsBig
TTimo: untested - that's the version from initial base64 decode
================
*/
int IntForSixtetsBig( byte *in ) {
	int ret = 0;
	ret |= in[0];
	ret |= in[1] << 6;
	ret |= in[2] << 2*6;
	ret |= in[3] << 3*6;
	return ret;
}

/*
================
Swap_Init
================
*/
void Swap_Init()
{
	byte	swaptest[2] = {1,0};

	// set the byte swapping variables in a portable manner	
	if ( *(short *)swaptest == 1) {
		// little endian ex: x86
		_BigShort = ShortSwap;
		_LittleShort = ShortNoSwap;
		_BigLong = LongSwap;
		_LittleLong = LongNoSwap;
		_BigFloat = FloatSwap;
		_LittleFloat = FloatNoSwap;
		_BigRevBytes = RevBytesSwap;
		_LittleRevBytes = RevBytesNoSwap;
		_LittleBitField = RevBitFieldNoSwap;
		_SixtetsForInt = SixtetsForIntLittle;
		_IntForSixtets = IntForSixtetsLittle;
	} else {
		// big endian ex: ppc
		_BigShort = ShortNoSwap;
		_LittleShort = ShortSwap;
		_BigLong = LongNoSwap;
		_LittleLong = LongSwap;
		_BigFloat = FloatNoSwap;
		_LittleFloat = FloatSwap;
		_BigRevBytes = RevBytesNoSwap;
		_LittleRevBytes = RevBytesSwap;
		_LittleBitField = RevBitFieldSwap;
		_SixtetsForInt = SixtetsForIntBig;
		_IntForSixtets = IntForSixtetsBig;
	}
}

/*
==========
Swap_IsBigEndian
==========
*/
bool Swap_IsBigEndian()
{
	byte	swaptest[2] = {1,0};
	return *(short *)swaptest != 1;
}
