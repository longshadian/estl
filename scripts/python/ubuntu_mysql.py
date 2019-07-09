#!/usr/bin/python3 -B
import mysql.connector as MySql

import random

def create_connection(user, password, host, database):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, buffered=False)
    except MySql.Error as err:
        print(err)

def db_query_room(conn):
    room_id_list = []
    try:
        cursor = conn.cursor()
        sql = "select room_id, name, sort from zww_room"
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            room_id_list.append(row[0])
            #print("{room_id} {name} {sort}".format(room_id=row[0], name=row[1], sort=row[2]))
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
            sort = random.randint(10, 1000000);
            print("room:{} sort:{}".format(room_id, sort))
            sql = "update zww_room set `sort` = %(sort)s where room_id = %(room_id)s;"
            cursor.execute(sql, { "sort":sort, "room_id":room_id})
        conn.commit()
        cursor.close()
    except MySql.Error as err:
        print(err)
    return False

def main():
    conn = create_connection(user="root", password="123456", host="192.168.0.242", database="zww")
    if conn:
        room_id_list = db_query_room(conn)
        db_update_room_sort(conn, room_id_list=room_id_list)
        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
