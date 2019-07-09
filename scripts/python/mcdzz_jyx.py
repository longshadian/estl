#!/usr/bin/python3 -B
import mysql.connector as MySql

import random

def create_connection(user, password, host, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, buffered=False)
    except MySql.Error as err:
        print(err)

def select_battle_log(conn, cnt, tm):
    try:
        cursor = conn.cursor()
        ## 出错
        #sql = "select count(cnt) from (select count(user_id) as cnt from battle_log where date(createtime) = %(tm)s group by user_id) as x where x.cnt = %(cnt)s; "
        sql = "select count(cnt) from (select count(user_id) as cnt from battle_log where date(createtime) = '{tm}' group by user_id) as x where x.cnt = {idx}".format(tm=tm,idx=cnt)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print(row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_battle_log_greater(conn, cnt, tm):
    try:
        cursor = conn.cursor()
        sql = "select count(cnt) from (select count(user_id) as cnt from battle_log where date(createtime) = '{tm}' group by user_id) as x where x.cnt > {idx}".format(tm=tm,idx=cnt)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print(">{}".format(cnt), row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_battle_log_total(conn, tm):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(distinct(user_id)) FROM mcdzz.battle_log where date(createtime) = '{tm}'".format(tm=tm)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("总人数", row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_battle_log_total_count(conn, tm):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM mcdzz.battle_log where date(createtime) = '{tm}'".format(tm=tm)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("总次数", row[0])
    except MySql.Error as err:
        print(err)
    return False

def selectStatisUser(conn, openidList = []):

    try:
        cursor = conn.cursor()
        for openid in openidList:
            openid = openid[:-1]
            #sql = "SELECT register_tm, login_tm FROM statis_user where openid = %(openid)s"
            sql = "INSERT INTO `temp_openid1128` (`openid`) VALUES (%(openid)s)"
            cursor.execute(sql, {"openid": openid})
            conn.commit()
            print(openid)
        print(len(openidList))
    except MySql.Error as err:
        print(err)
    return False

def readOpenid():
    fpath = r"Z:\temp\mcdzz1128\x.value"
    with open(fpath, 'r') as f:
        l = f.readlines()
        return l


def main():
    conn = create_connection(user="root", password="Buzhidao123*", host="120.27.250.213", database="mcdzz")
    if conn:
        openidList = readOpenid()
        selectStatisUser(conn, openidList)
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
