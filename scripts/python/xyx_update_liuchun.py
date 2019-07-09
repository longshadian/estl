#!/usr/bin/python3 -B
import mysql.connector as MySql

import random

'''
小游戏更新留存，
从statis_total表中获取statistime和channel，重新计算留存 
'''



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
        sql = "select count(cnt) from (select count(user_id) as cnt from answer_log where date(createtime) = '{tm}' group by user_id) as x where x.cnt = {idx}".format(tm=tm,idx=cnt)
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
        sql = "select count(cnt) from (select count(user_id) as cnt from answer_log where date(createtime) = '{tm}' group by user_id) as x where x.cnt > {idx}".format(tm=tm,idx=cnt)
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
        sql = "SELECT count(distinct(user_id)) FROM answer_log where date(createtime) = '{tm}'".format(tm=tm)
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
        sql = "SELECT count(id) FROM answer_log where date(createtime) = '{tm}'".format(tm=tm)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("总次数", row[0])
    except MySql.Error as err:
        print(err)
    return False

def table_name_yyyymmdd(daily):
    vector = []
    for val in str(daily).split('-'):
        vector.append(int(val))
    return "{}{}{%0}".format(vector[0],vector[1],vector[2])



def main():
    conn = create_connection(user="root", password="Buzhidao123*", host="120.27.250.213", database="sstc")
    if conn:
        daily = "2018-11-11"
        maxIdx = 20
        for idx in range(1, maxIdx + 1):
            select_battle_log(conn, idx, daily)
        select_battle_log_greater(conn, maxIdx, daily)
        select_battle_log_total(conn, daily)
        select_battle_log_total_count(conn, daily)
        conn.close()

if __name__ == "__main__":
    #main()
    x ="aaa_{0}{1:0>2}".format(2018,10)
    print(x)
