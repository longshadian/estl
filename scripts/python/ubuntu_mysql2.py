#!/usr/bin/python3 -B
import mysql.connector as MySql

import datetime

def db_insert_tcpp(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("delete from tcpp")
        conn.commit()

        for i in range(0, 2):
            #sql = "insert into tcpp (fnum1, fnum2, fstr1) values (%s, %s, %s)"
            #cursor.execute(sql, (1,2,"ff"))
            sql = "insert into tcpp (`fstr1`, `fdatetime`, `%`, `fbigint`) values (%(fstr1)s, %(fdatetime)s, %(%)s, %(fbigint)s)"
            d = datetime.datetime.now()
            cursor.execute(sql, { "fstr1":"哈哈", "fdatetime":d, "%": 1112, "fbigint":1234567890123456789})
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except MySql.Error as err:
        print(err)
    return False


def db_query_tcpp(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("select fstr1, fdatetime,fbigint, `%` from tcpp")
        rows = cursor.fetchall()
        for row in rows:
            print(row[0])

        #for (fstr1, fdatetime, fbigint, a) in cursor:
        #    print("{0} {1} {2} {3}".format(fstr1, fdatetime, fbigint, a))
        cursor.close()
        conn.close()
        return True
    except MySql.Error as err:
        print(err)
    return False



def create_connection(user, password, host, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, buffered=False)
    except MySql.Error as err:
        print(err)


def db_insert_tcpp_ex(conn):
    cursor = conn.cursor()
    with open("./3.csv", encoding='utf_8_sig') as f:
        rows = csv.reader(f)
        for row in rows:
            sql = "insert into tcpp (`fstr1`, `fstr2`, `fstr3`) values (%s, %s, %s)"
            cursor.execute(sql, row)
    conn.commit()
    cursor.close()
    conn.close()


def db_query_tuser(conn):
    try:
        cursor = conn.cursor()
        sql = "select fid, fgameid from {0}".format("tapk")
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("{fid}    {fgameid}".format(fid = row[0], fgameid = row[1]))
        cursor.close()
        conn.close()
        return True
    except MySql.Error as err:
        print(err)
    return False


def main():
    conn = create_connection(user="root", password="123456", host="192.168.3.3", database="mytest")
    if conn:
        db_query_tuser(conn)

if __name__ == "__main__":
    main()
