#!/usr/bin/python3 -B
import mysql.connector as MySql

import random
import datetime

'''
每天汇总统计gamebox.click_statistic click_count总数

'''

def create_connection(user, password, host, port, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, port=port, buffered=False)
    except MySql.Error as err:
        print(err)

def db_query_click_statistic(conn, date):
    total = 0
    try:
        cursor = conn.cursor()
        sql = "select sum(click_count) from click_statistic where date(createtime) = %(date)s;"
        cursor.execute(sql, {"date":date})
        rows = cursor.fetchall()
        for row in rows:
            total = row[0]
        cursor.close()
        return total
    except MySql.Error as err:
        print(err)
    return -1

def db_insert_click_statistic_total(conn, count, date):
    try:
        cursor = conn.cursor()
        sql = "INSERT INTO `click_statistic_total` (`datetime`, `count`, `createtime`) values (%(date)s, %(count)s, NOW())"
        cursor.execute(sql, {"date": date, "count": count})
        conn.commit()
        cursor.close()
    except MySql.Error as err:
        print(err)


def getYesterday():
    today = datetime.date.today()
    oneday = datetime.timedelta(days=1)
    return (today-oneday).isoformat()

def main():
    tm = datetime.date(2018, 8, 12).isoformat()
    #tm = getYesterday()

    '''
    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.105.106", port=3306, database="gamebox")
    if conn:
        total = db_query_click_statistic(conn, tm)
        db_insert_click_statistic_total(conn, total, tm)
        conn.close()

    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.105.106", port=3306, database="gamebox1")
    if conn:
        total = db_query_click_statistic(conn, tm)
        db_insert_click_statistic_total(conn, total, tm)
        conn.close()

    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.105.106", port=3306, database="gamebox2")
    if conn:
        total = db_query_click_statistic(conn, tm)
        db_insert_click_statistic_total(conn, total, tm)
        conn.close()
        
    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.105.106", port=3306, database="gamebox_v2")
    if conn:
        total = db_query_click_statistic(conn, tm)
        db_insert_click_statistic_total(conn, total, tm)
        conn.close()
        '''
    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.105.106", port=3306, database="gamebox_v2_yqwxyx")
    if conn:
        total = db_query_click_statistic(conn, tm)
        db_insert_click_statistic_total(conn, total, tm)
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    tm = datetime.datetime.today().isoformat(sep=' ')
    print(tm)
    #main()
