#ifndef _RATE_GENERATOR_H_
#define _RATE_GENERATOR_H_

#include <vector>

//独立事件概率生产器，计算加权求平均
// struct item
// {
//     item(int id, int v) : index(id), val(v) {}
//     int rates() const { return val; }
//     int index;
//     int val;
// };
//
// item m1(1, 100);
// item m2(2, 200);
// item m3(3, 100);
// item m4(4, 100);
// 
// RateGenarator<item> g;
// g.putRes(&m1, &item::rates);
// g.putRes(&m2, &item::rates);
// g.putRes(&m3, &item::rates);
// g.putRes(&m4, &item::rates);
// 
// int n = 1000000;
// std::map<int, int> all;
// while (n--) { all[g.randShuffle()->name]++; }
// -----------------------------------------------------
//  1 : 201201
//  2 : 401962 
//  3 : 198241 
//  4 : 198596

template <typename T>
class RateGenarator
{
    class RateWrapper
    {
    public:
        RateWrapper(int min, int max, const T* p) : mCloseMin(min), mOpenMax(max), mRes(p) { }
        ~RateWrapper() {}

        int closeMin() const { return mCloseMin; }
        int openMax() const { return mOpenMax; }
        const T* res() const { return mRes; }
    private:
        int mCloseMin;
        int mOpenMax;
        const T* mRes;
    };
public:
    RateGenarator() : mRangeDivisor(0), mRanges() { }

    template<typename It, typename F>
    RateGenarator(It b, It e, F ptr) : mRangeDivisor(0), mRanges() 
    { 
        for (; b != e; ++b) {
            putRes(*b, ptr);
        }
    }

    ~RateGenarator() { for (size_t i = 0; i != mRanges.size(); ++i) delete mRanges[i]; }

    void putRes(const T* pItem, int (T::*ptr)() const)
    {
        int val = (pItem->*ptr)();
        mRanges.push_back(new RateWrapper(mRangeDivisor, mRangeDivisor + val, pItem));
        //std::cout << val << std::endl;
        mRangeDivisor += val;
    }

    const T* randShuffle() const
    {
        if (mRangeDivisor <= 0)
            return NULL;
        int num = rand()%mRangeDivisor;

        for (size_t i = 0; i != mRanges.size(); ++i) {
            const RateWrapper* pWrapper = mRanges[i];
            if (pWrapper->closeMin() <= num && num < pWrapper->openMax()) {
                return pWrapper->res();
            }
        }
        return NULL;
    }
private:
    int                         mRangeDivisor;
    std::vector<RateWrapper*>   mRanges;
};

#endif
