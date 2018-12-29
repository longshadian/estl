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

def call_sp(conn, name, tm):
    try:
        #cursor = conn.cursor()
        sql = "call sp_statis_jyx ('{name}', '{tm}'); ".format(name=name, tm=tm)
        print(sql)
        #cursor.execute(sql)
        return True
    except MySql.Error as err:
        print(err)
    return False

def tm_range(tm1, tm2):
    d1 = datetime.datetime(2018, 11, 1)
    d2 = datetime.datetime(2018, 11, 30)
    d = d1
    delta = datetime.timedelta(days=1)
    while d <= d2:
        print(d.strftime("%Y-%m-%d"))
        d += delta

def main():
    conn = create_connection(user="root", password="Buzhidao123*", host="120.27.250.213", database="qmbk")
    if conn:
        name = 'gamebox_v2'
        d1 = datetime.datetime(2018, 9, 27)
        d2 = datetime.datetime(2018, 11, 20)
        d = d1
        delta = datetime.timedelta(days=1)
        while d <= d2:
            tm = d.strftime("%Y-%m-%d")
            call_sp(conn, name, tm)
            d += delta
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
