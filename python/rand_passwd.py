import random

def randStr(len = 8):
    src = []
    src.extend(range(ord('0'), ord('9')))
    src.extend(range(ord('a'), ord('z')))
    src.extend(range(ord('A'), ord('Z')))
    random.shuffle(src)

    src = src[:len]
    ret = ""
    for v in src:
        ret += chr(v)
    return ret

def main():
    random.seed()
    for i in range(1, 10):
        print(randStr(20))

if __name__ == '__main__':
    main()