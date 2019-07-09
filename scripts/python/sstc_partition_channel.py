#!/usr/bin/python3 -B
import mysql.connector as MySql

import random
import datetime
import time

def create_connection(user, password, host, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, buffered=False)
    except MySql.Error as err:
        print(err)

def select_answer_log_partition(conn, cnt, tm, channel):
    try:
        cursor = conn.cursor()
        sql = '''
select count(cnt) from (
	SELECT count({tb}.user_id) as cnt FROM {tb} 
		inner join user on {tb}.user_id = user.user_id
		inner join statis_user on user.openid = statis_user.openid
		where statis_user.channel = '{channel}'
		group by {tb}.user_id
    ) as x where x.cnt = {idx};
'''
        sql = sql.format(tb=table_name(tm), tm=tm, idx=cnt, channel=channel)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print(row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_answer_log_greater_partition(conn, cnt, tm, channel):
    try:
        cursor = conn.cursor()
        sql = '''
select count(cnt) from (
	SELECT count({tb}.user_id) as cnt FROM {tb} 
		inner join user on {tb}.user_id = user.user_id
		inner join statis_user on user.openid = statis_user.openid
		where statis_user.channel = '{channel}'
		group by {tb}.user_id
    ) as x where x.cnt > {idx};
'''
        sql = sql.format(tb=table_name(tm), tm=tm, idx=cnt, channel=channel)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print(">{}".format(cnt), row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_answer_log_total_partition(conn, tm, channel):
    try:
        cursor = conn.cursor()
        sql = '''
SELECT count( distinct( {tb}.user_id)) FROM {tb} 
	inner join user on {tb}.user_id = user.user_id
	inner join statis_user on user.openid = statis_user.openid
	where statis_user.channel = '{channel}'
'''
        sql = sql.format(tb=table_name(tm), tm=tm, channel=channel)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("总人数", row[0])
    except MySql.Error as err:
        print(err)
    return False

def select_answer_log_total_count_partition(conn, tm, channel):
    try:
        cursor = conn.cursor()
        sql = '''
SELECT count(  {tb}.id) FROM {tb} 
	inner join user on {tb}.user_id = user.user_id
	inner join statis_user on user.openid = statis_user.openid
	where statis_user.channel = '{channel}'
'''
        sql = sql.format(tb=table_name(tm), tm=tm, channel=channel)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            print("总次数", row[0])
    except MySql.Error as err:
        print(err)
    return False


def table_name(daily):
    vector = []
    for val in str(daily).split('-'):
        vector.append(val)
    return "answer_log_{}{}{}".format(vector[0],vector[1],vector[2])


def main():
    conn = create_connection(user="root", password="Buzhidao123*", host="120.27.250.213", database="sstc")
    if conn:
        daily = "2018-12-06"
        channel = 'krq_sstc=149'
        maxIdx = 20
        for idx in range(1, maxIdx + 1):
            select_answer_log_partition(conn, idx, daily, channel)
        select_answer_log_greater_partition(conn, maxIdx, daily, channel)
        select_answer_log_total_partition(conn, daily, channel)
        select_answer_log_total_count_partition(conn, daily, channel)
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
