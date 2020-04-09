#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <list>
#include <functional>

/// 一张扑克牌用int32_t表示。只使用最低8位
/// 低8位描述, 高4位代表花色，低4位代表牌值
/// 数值	   0        1       2       3	
/// 花色	 方块     梅花      红桃  	黑桃	

/// 数值	    0       1   2   3   4   5   6   7   8   9   10  11  12  13  14      15
/// 牌值	    空牌    3   4   5   6   7   8   9   10  J   Q   K   A   2   小王	    大王
/// 小王和大王高4位设为0

namespace gp_alg {

/// 花色
enum class Color
{
    Diamond  = 0,       // 方块
    Club     = 1,       // 梅花
    Heart    = 2,       // 红桃
    Spade    = 3        // 黑桃
};

/// 牌值
enum class Value
{
    Card_Null   = 0,
    Card_3      = 1,
    Card_4      = 2,
    Card_5      = 3,
    Card_6      = 4,
    Card_7      = 5,
    Card_8      = 6,
    Card_9      = 7,
    Card_10     = 8,
    Card_J      = 9,
    Card_Q      = 10,
    Card_K      = 11,
    Card_A      = 12,
    Card_2      = 13,
    B_Joker     = 14,
    R_Joker     = 15,
    Card_Count
};

/// 牌型
enum class Type 
{
    Type_Null          = 0x00,	//未定义
    Type_Signle        = 0x01,	//单牌
    Type_Single_Ser    = 0x02,	//顺子
    Type_Double        = 0x03,	//对子
    Type_Double_Ser    = 0x04,	//连对
    Type_Triple        = 0x05,	//三个
    Type_Triple_Ser    = 0x06,	//三顺(飞机)
    Type_31            = 0x07,	//3+1
    Type_31_Ser        = 0x08, //3+1飞机
    Type_32            = 0x09,	//3+2
    Type_32_Ser        = 0x0A, //3+2飞机
    //CARD_TYPE_41            = 0x0B, //4带1
    Type_411           = 0x0B,	//4带2单
    Type_422           = 0x0C,	//4带2对
    Type_Bomb          = 0x0D,	//炸弹
    //CARD_TYPE_BOMB_SER      = 0x0E,	//炸弹
    Type_Rocket        = 0x0E,	//火箭
    Type_Mask          = 0x10,	//牌型掩码
};

/// 卡牌数量
enum { QUANTITY = 54 };

using Card = int32_t;

using CardPosition = std::list<Card>;    //牌所在的位置, size()最大为4

struct CardType
{
    CardType() 
        : m_type(Type_Null), m_value(Card_Null), m_type_len(0)
    {}

    CardType(Type type, int32_t value, int32_t type_len) 
        : m_type(type), m_value(value), m_type_len(type_len)
    {}

     //牌型
    int32_t     m_type;

    //牌值:取得是规则化后主牌的最小值，例如:
    //JJJ QQQ KKK 33 44 55,这手牌的牌值是J
    // A K J Q 10 7 8 9,规则化后AKQJ10987 这牌值是7
    int32_t     m_value;

    //牌型长度(如果该牌型有长度)
    int32_t     m_type_len;
};

inline
int32_t makeColor(Card card, Color color)
{
    return (((card) & 0x0F) | (color) << 4)
}

inline
int32_t makeValue(int32_t card, Value value)
{
    return (((card) & 0xF0) | (value))
}

inline 
Color getCardColor(int32_t card) 
{
    return (((card)&0x30) >> 4)
}

inline 
Value getCardValue(int32_t card)
{
    return ((card) & 0x0F)
}

/*
#define setCardColor(card, color)   (((card) & 0x0F) | (color) << 4)
#define setCardValue(card, value)	(((card) & 0xF0) | (value))
#define	getCardColor(card)			(((card)&0x30) >> 4)
#define	getCardValue(card)			((card) & 0x0F)
#define setCard(color,value)		(((color)<<4) | ((value)&0x0F))
*/

/** 
 * 初始化牌型
 */
void initCard(std::vector<Card>& card);
std::vector<Card> initCard();

/**
 * 检查选中牌能否压过目标牌,如果src为空或src_len==0，只要检查选中牌是否合法
 * 注意：src hand无需牌型化
 * Params   src,src_len 目标牌
 *          hand,hand_len 选中的牌
 * Returns  true:能压过    false:不能压过
 */
bool check(const Card* src,  int32_t src_len,
           const Card* hand, int32_t hand_len);

/**
 * 检查选中牌能否压过目标牌,如果src为空或src_len==0，只要检查选中牌是否合法
 * 注意：src hand无需牌型化
 * Params   src,src_len 目标牌
 *          selected,selected_len 选中的牌
 *          out:打出去的牌而且是规则化的。
 *          注意:out缓冲区不能小于selected_len
 * Returns  true:能压过    false:不能压过
 */
bool playCard(const Card* src,       int32_t src_len,
              const Card* selected,  int32_t selected_len,
              Card* out);

/**
 * 自动选择选中可以压过目标的牌。类似提示功能
 * Params   src,src_len:目标牌
 *          hand,hand_len:手中的牌
 *          out:值不等于0就是选中的能压过目标牌的牌。
 *          注意：out的长度不能少于hand_len,选中的牌是未规则化。
 * Returns  true:能压过    false:不能压过
 */
bool autoSelect(const Card* src,     int32_t src_len,
                const Card* hand,    int32_t hand_len,
                Card* out); 
bool autoSelectEx(const Card* src,     int32_t src_len,
                const Card* hand,    int32_t hand_len,
                Card* out, CardType* out_type); 

/**
 * 判断出牌类型，输出规则化牌型
 * Params   src,src_len:需要判断的牌
 *          out:输出规则化牌型
 * Returns  返回牌型和牌型长度(如果该牌型有牌型长度)
 */
CardType parseCardType(const Card* src, int32_t len, Card* out);
CardType parseCardType(const Card* src, int32_t len);
CardType parseCardType(std::vector<Card>* src);

/**
 * 选中1张，或2张牌后会自动补全。src为空会选择对子,或顺子。src不为空会尝试去选比src大的顺子或对子
 * Params   src,src_len:目标牌(可以为空)
 *          selected,selected_len:选中的牌
 *          hand,hand_len:玩家手中的牌
 *          out:最终自动补全的牌
 *          注意：out的长度不能少于hand_len,选中的牌是未规则化。
 * Returns  返回自动补全是否成功
 */
bool autoComplete(const Card* src,       int32_t src_len,        
                  const Card* selected,  int32_t selected_len,
                  const Card* hand,      int32_t hand_len,
                  Card* out);
/**
 * 拖牌自动补齐
 * Params   selected,selected_len:拖牌选中的牌
 *          hand,hand_len:手中所有的牌
 *          out:自动补全的牌
 *          注意：out的长度不能少于hand_len,选中的牌是未规则化。
 * Return   返回拖牌自动补全是否成功
 */
bool autoSlide(const Card* selected, int32_t selected_len,
               const Card* hand,     int32_t hand_len,
               Card* out);

/**
* 自动出牌，出最小的牌，最小的如果是单牌，对子，炸弹，火箭就会出去
* 如果在不拆牌的情况下能组成顺子，连对，飞机，就会出。
* 这是一个比较简单的AI
* Params   hand,hand_len:手中所有的牌
* Return   返回出的牌,已经规则化好的
*/
std::vector<Card> autoPlayMinCards(const Card* hand, int32_t hand_len);
std::vector<Card> simpleAutoPlayMinCards(const Card* hand, int32_t hand_len);

//去除空牌
std::vector<Card> removeCardNull(const Card* src, int32_t len);
std::vector<Card> removeCardNull(const std::vector<Card>& src);

int32_t compareCardType(CardType type1, CardType type2);

std::vector<Card> sort_for_show(std::vector<Card> * src);

/**
*  获取所有炸弹,王炸
*/
std::vector<Card> bombCard(const Card* hand, int32_t hand_len);


//比较牌值大小和花色大小，默认降序
class PreGreaterSort : public std::binary_function<Card, Card, bool>
{
public:
    bool operator()(Card x, Card y) const
    {
        if (x == y)
            return false;
        int x_val = getCardValue(x);
        int y_val = getCardValue(y);

        if (y_val < x_val)
            return true;
        if (x_val < y_val)
            return false;

        //牌值相同按花色排序
        int x_color = getCardColor(x);
        int y_color = getCardColor(y);

        if (y_color < x_color)
            return true;
        if (x_color < y_color)
            return false;
        return true;
    }
};

class PreCardValue : public std::unary_function<Card, bool>
{
    int32_t val;
public:
    PreCardValue(int32_t v) : val(v) {}
    bool operator()(Card x) const
    {
        return val == getCardValue(x);
    }
};


/************************************************************************
 * 内部实现函数,请不要调用这些接口                                                                 
 ************************************************************************/
namespace detail {

bool pickupRocket(Card* src, Card* src_end, std::vector<Card>* out);

std::vector<Card> autoPlayMinCardsDetail(const Card* hand, int32_t hand_len);

std::vector<Card> exceptBombRocket(std::vector<Card> hand, std::vector<Card>* rocket, std::vector<std::vector<Card>>* bomb);
int32_t sameValueCount(const Card* hand, int32_t hand_len);

//按牌值从小到大，记录相同牌值所在的位置
void classifyHandCard(const Card* hand, int32_t hand_len, CardPosition* card_cls);

void sort(Card* src, int32_t len, bool desc);

int32_t compareCard(Card card1, Card card2);

int32_t getSameValueSeq(const Card* src, int32_t len, uint16_t* same_seq, bool need_sort);

void normalizeTypeList(uint16_t* same_seq, int32_t len, Card* out);

bool isSeries(const Card* src, int32_t len);
int32_t isSeries(const uint16_t* src, int32_t len, int32_t cnt);

//选择比card_val大的相同的牌count张 优先选择相同牌少的
bool searchAtomic(CardPosition* card_cls, Card card_val, int32_t count,
                  const Card* out_src, Card* out);

//选择顺子, 选择的顺子需要比min_card大, count：是单顺 双顺 三顺
bool searchSeries(CardPosition* card_cls, Card src_min_card, Card src_max_card, int32_t src_series_count,
                  const Card* out_src, Card* out);

bool searchBomb(CardPosition* card_cls, Card card_val, const Card* out_src, Card* out);
//bool searchBombSer(CardPosition* card_cls, Card card_val, int32_t ser_count, const Card* out_src, Card* out);

//选择2张单牌，不能包含火箭
//bool searchTwoSingleExceptRocket(CardPosition* card_cls, const Card* out_src, Card* out);

//选择2手对子，不能
bool searchTwoDoubleExceptBomb(CardPosition* card_cls, const Card* out_src, Card* out);
} //detail

/************************************************************************
 * 工具函数                                                                 
 ************************************************************************/
namespace utility {

struct CardNumString
{
    int32_t     m_num;
    std::string m_string;
};

std::string printCards(const Card* src, int32_t len);
std::string printCards(const std::vector<Card>& src);
std::string printCardValue(const std::vector<Card>& src);

std::string cardTypeToString(int32_t type);
std::string cardValueToString(int32_t value);
int32_t     stringToCardValue(std::string s);

//从src中选取牌值为value的n张填充至out
void selectCards(std::vector<Card>* src, int32_t value, std::vector<Card>* out, int32_t n = 1);

//从src中选取最多n张填充至out
void paddingCardsCount(std::vector<Card>* src, int32_t n, std::vector<Card>* out);

//把字符串转成牌
std::vector<Card> pickCardsFromString(std::string s, std::vector<Card>* all_cards);

} // utility

} // gp_alg 
