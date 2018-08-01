
#检查csv中的牌型是否正确
import csv


def read_csv(path):
    try:
        file = open(path, "r");
        reader = csv.reader(file)
        ret = []
        pos = 1
        for row in reader:
            ret.append([row, pos])
            pos += 1
        for row in ret:
            pass
            #print("%4d %s %s %s %s" % (row[1], row[0][0], row[0][1], row[0][2], row[0][3]))
        return ret
    except IOError:
        print("open error %s " % (path))


def pickup(cards, val):
    if val in cards:
        idx = cards.index(val)
        cards.pop(idx)
        return True
    return False


def select_card(row):
    if len(row[0][0]) != 3 or len(row[0][1]) != 17 or len(row[0][2]) != 17 or len(row[0][3]) != 17:
        print("error: row:%d len" % (row[1]))
        return

    base_cards = ['3', '3', '3', '3',
                  '4', '4', '4', '4',
                  '5', '5', '5', '5',
                  '6', '6', '6', '6',
                  '7', '7', '7', '7',
                  '8', '8', '8', '8',
                  '9', '9', '9', '9',
                  'T', 'T', 'T', 'T',
                  'J', 'J', 'J', 'J',
                  'Q', 'Q', 'Q', 'Q',
                  'K', 'K', 'K', 'K',
                  'A', 'A', 'A', 'A',
                  '2', '2', '2', '2',
                  'B', 'R']

    for c in row[0][0]:
        if not pickup(base_cards, c):
            print("error: row:%d bottom %s" % (row[1], c))
            return

    for c in row[0][1]:
        if not pickup(base_cards, c):
            print("error: row:%d card_1 %s" % (row[1], c))
            return

    for c in row[0][2]:
        if not pickup(base_cards, c):
            print("error: row:%d card_2 %s" % (row[1], c))
            return

    for c in row[0][3]:
        if not pickup(base_cards, c):
            print("error: row:%d card_2 %s" % (row[1], c))
            return

    if len(base_cards) != 0:
        print("error: row:%d remain cards" % (row[1]))


def main(path):
    rows = read_csv(path)
    for row in rows:
        select_card(row)

if __name__ == '__main__':
    #main(r'Y:\ddzgame\src2\tools\card_import\6.csv')
    main(r'/home/cgy/work/ddzgame/src2/tools/card_import/0.csv')
    