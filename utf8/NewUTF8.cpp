

/* Bytes Bits Hex Min  Hex Max  Byte Sequence in Binary */
/*   1     7  00000000 0000007f 0vvvvvvv */
/*   2    11  00000080 000007FF 110vvvvv 10vvvvvv */
/*   3    16  00000800 0000FFFF 1110vvvv 10vvvvvv 10vvvvvv */
/*   4    21  00010000 001FFFFF 11110vvv 10vvvvvv 10vvvvvv 10vvvvvv */
/*   5    26  00200000 03FFFFFF 111110vv 10vvvvvv 10vvvvvv 10vvvvvv 10vvvvvv */
/*   6    31  04000000 7FFFFFFF 1111110v 10vvvvvv 10vvvvvv 10vvvvvv 10vvvvvv 10vvvvvv */

struct Tab
{
    int cmask;
    int cval;
    int shift;
    long    lmask;
    long    lval;
};

static Tab tab[] =
{
    0x80,   0x00,   0 * 6,    0x7F,       0,          /* 1 byte sequence */
    0xE0,   0xC0,   1 * 6,    0x7FF,      0x80,       /* 2 byte sequence */
    0xF0,   0xE0,   2 * 6,    0xFFFF,     0x800,      /* 3 byte sequence */
    0xF8,   0xF0,   3 * 6,    0x1FFFFF,   0x10000,    /* 4 byte sequence */
    0xFC,   0xF8,   4 * 6,    0x3FFFFFF,  0x200000,   /* 5 byte sequence */
    0xFE,   0xFC,   5 * 6,    0x7FFFFFFF, 0x4000000,  /* 6 byte sequence */
    0,                                              /* end of table */
};

/* s 指向 UTF-8 字节序列，n 表示字节长度 */
/* p 指向一个 wchar_t 变量 */
/* mbtowc 对 s 指定的字节进行解码，得到的 Unicode 存到 p 指向的变量 */
int mbtowc(wchar_t *p, char *s, size_t n)
{
    long l;
    int c0, c, nc;
    Tab *t;

    if (s == 0)
        return 0;

    nc = 0;
    if (n <= nc)
        return -1;

    /* c0 保存第一个字节内容，后面会移动 s 指针，此处备份一下 */
    /* 汉字「吕」的 UTF-8 编码是 `11100101`, `10010000`, `10010101` */
    /* 此时 l = c0 = 11100101 */
    c0 = *s & 0xff;
    /* l 保存 Unicode 结果 */
    l = c0;

    /* 根据 UTF-8 的表示范围从小到大依次检查 */
    for (t = tab; t->cmask; t++) {
        /* nc 以理解为 tab 的行号 */
        /* tab 行号跟这个范围内 UTF-8 编码所需字节数量相同 */
        nc++;

        /* c0 指向第一个字节，不会变化 */
        /* l 在 n == 1 和 n == 2 时左移 6 位两次 */
        /* 到 nc == 3 时才会进入该分支 */
        /* 此时的 l 已经是 11100101+010000+010101 了 */
        if ((c0 & t->cmask) == t->cval) {
            /* lmaxk 表示三字节能表示的 Unicode 最大值 */
            /* 使用 & 操作，移除最高位的 111 */
            /* 所以 l 最终为 00000101+010000+010101 */
            /* 也就是 l = 0x5415，对应 Unicode +U5415 */
            l &= t->lmask;

            if (l < t->lval)
                return -1;

            /* 保存结果并反回 */
            *p = l;
            return nc;
        }

        if (n <= nc)
            return -1;

        /* s 指向下一个字节 */
        s++;
        /* 0x80 = 10000000 */
        /* UTF-8 编码从第二个字节开始高两位都是 10 */
        /* 这一步是为了把最高位的 1 去掉 */
        c = (*s ^ 0x80) & 0xFF;
        /* n == 1 时 */
        /* c = 00010000 */
        /* n == 2 时 */
        /* c = 00010101 */

        /* 0xc0 = 1100000 */
        /* 这一上检查次高位是否为 1，如果是 1，则为非法 UTF-8 序列 */
        if (c & 0xC0)
            return -1;

        /* c 只有低 6 位有效 */
        /* 根据 UTF-8 规则，l 左移 6 位，将 c 的低 6 位填入 l */
        l = (l << 6) | c;
        /* n == 1 时 */
        /* l 的值变成 11100101+010000 */
        /* n == 2 时 */
        /* l 的值变成 11100101+010000+010101 */
    }
    return -1;
}

