#!/usr/bin/python3 -B
import mysql.connector as MySql

import random

def create_connection(user, password, host, port, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, port=port, buffered=False)
    except MySql.Error as err:
        print(err)

def db_query_room(conn):
    room_id_list = []
    try:
        cursor = conn.cursor()
        sql = "select room_id, sort from zww_room"
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            room_id_list.append(row[0])
            #print("{room_id} {sort}".format(room_id=row[0], sort=row[1]))
        cursor.close()
        return room_id_list
    except MySql.Error as err:
        print(err)
    return room_id_list

def db_update_room_sort(conn, room_id_list = []):
    if len(room_id_list) == 0:
        return 0
    try:
        cursor = conn.cursor()
        for room_id in room_id_list:
            if room_id == 100 or room_id == 101:
                continue
            sort = random.randint(10, 1000000);
            print("room:{} sort:{}".format(room_id, sort))
            sql = "update zww_room set `sort` = %(sort)s where room_id = %(room_id)s;"
            cursor.execute(sql, { "sort":sort, "room_id":room_id})
        conn.commit()
        cursor.close()
    except MySql.Error as err:
        print(err)
    return False

def db_update_room_sort_group(conn):
    try:
        cursor = conn.cursor()
        sql = "update zww_room_sort set sort_value = CEIL(RAND()*1000000) where id > 0 and room_id != 100 and room_id != 101;"
        cursor.execute(sql)
        conn.commit()
        cursor.close()
    except MySql.Error as err:
        print(err)
    return False

def main():
    #conn = create_connection(user="root", password="Buzhidao123*", host="120.27.250.213", port=3306, database="zww")
    conn = create_connection(user="root", password="zww#Test2018", host="114.55.104.197", port=3810, database="zww")
    if conn:
        room_id_list = db_query_room(conn)
        db_update_room_sort(conn, room_id_list=room_id_list)
        db_update_room_sort_group(conn)
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
