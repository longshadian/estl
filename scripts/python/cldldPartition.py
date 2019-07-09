#!/usr/bin/python3 -B
import mysql.connector as MySql

import random
import datetime
import time

def create_connection(user, password, host, database, port):
    try:
        return MySql.connect(user=user, password=password, host=host, database=database, buffered=False, port=port)
    except MySql.Error as err:
        print(err)


def TableName(daily):
    vector = []
    for val in str(daily).split('-'):
        vector.append(val)
    return "battle_log_{}{}{}".format(vector[0],vector[1],vector[2])


## 总次数结算
def SelectZhongCishuAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where achieve = 1;".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 总次数结算
def SelectZhongCishuTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb}".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 机器人结算
def SelectRobotAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where is_robot = 1 and achieve = 1;".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 机器人总次数
def SelectRobotTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where is_robot = 1;".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 好友对战结算
def SelectInviteAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where type = 1 and achieve = 1;".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 好友对战总次数
def SelectInviteTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where type = 1;".format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1


## 金币场对战结算
def SelectRoomAchieve(conn, table_name, room_id):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where room_id = {room_id} and achieve = 1;".format(tb=table_name, room_id=room_id)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 金币场对战总次数
def SelectRoomTotal(conn, table_name, room_id):
    try:
        cursor = conn.cursor()
        sql = "SELECT count(id) FROM {tb} where room_id = {room_id};".format(tb=table_name, room_id=room_id)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

###########################################################
###########################################################
###########################################################

## 总次数结算
def SelectUserZhongCishuAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql='''
select count(distinct(uid)) from	
(select user_id as uid from {tb} where achieve = 1
	union all
select target_uid as uid from {tb} where achieve = 1
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 总次数结算
def SelectUserZhongCishuTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from	
(select user_id as uid from {tb}
    union all
select target_uid as uid from {tb}
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 机器人结算
def SelectUserRobotAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from
(select user_id as uid from {tb} where achieve = 1 and is_robot = 1
    union all
select target_uid as uid from {tb} where achieve = 1 and is_robot = 1
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 机器人总次数
def SelectUserRobotTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from	
(select user_id as uid from {tb} where is_robot = 1
    union all
select target_uid as uid from {tb} where is_robot = 1
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 好友对战结算
def SelectUserInviteAchieve(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from	
(select user_id as uid from {tb} where type = 1 and achieve = 1
    union all
select target_uid as uid from {tb} where type = 1 and achieve = 1
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 好友对战总次数
def SelectUserInviteTotal(conn, table_name):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from	
(select user_id as uid from {tb} where type = 1
    union all
select target_uid as uid from {tb} where type = 1
) as cp where uid != 0;'''.format(tb=table_name)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1


## 金币场对战结算
def SelectUserRoomAchieve(conn, table_name, room_id):
    try:
        cursor = conn.cursor()
        sql = '''
select count(distinct(uid)) from	
(select user_id as uid from {tb} where room_id = {room_id} and achieve = 1
    union all
select target_uid as uid from {tb} where room_id = {room_id} and achieve = 1
) as cp where uid != 0;'''.format(tb=table_name, room_id=room_id)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1

## 金币场对战总次数
def SelectUserRoomTotal(conn, table_name, room_id):
    try:
        cursor = conn.cursor()
        sql = '''
        select count(distinct(uid)) from	
        (select user_id as uid from {tb} where room_id = {room_id}
            union all
        select target_uid as uid from {tb} where room_id = {room_id}
        ) as cp where uid != 0;'''.format(tb=table_name, room_id=room_id)
        cursor.execute(sql)
        rows = cursor.fetchall()
        for row in rows:
            return row[0]
    except MySql.Error as err:
        print(err)
    finally:
        cursor.close()
    return -1




def main():
    #conn = create_connection(user="root", password="Buzhidao1234", host="rm-bp157g12pc08eu7tq.mysql.rds.aliyuncs.com", database="cldld")
    conn = create_connection(user="root", password="Buzhidao1234", host="114.55.104.197", port=7306, database="cldld")
    if conn:
        daily = "2019-01-19"
        table_name = TableName(daily)
        v1 = SelectZhongCishuAchieve(conn, table_name)
        v2 = SelectZhongCishuTotal(conn, table_name)
        print("总次数, {v1}, {v2}".format(v1=v1, v2=v2))

        v1 = SelectRobotAchieve(conn, table_name)
        v2 = SelectRobotTotal(conn, table_name)
        print("robot, {v1}, {v2}".format(v1=v1, v2=v2))

        v1 = SelectInviteAchieve(conn, table_name)
        v2 = SelectInviteTotal(conn, table_name)
        print("invite, {v1}, {v2}".format(v1=v1, v2=v2))

        for room_id in range(1, 5+1):
            v1 = SelectRoomAchieve(conn, table_name, room_id)
            v2 = SelectRoomTotal(conn, table_name, room_id)
            print("room, {v1}, {v2}".format(v1=v1, v2=v2))

####################
        v1 = SelectUserZhongCishuAchieve(conn, table_name)
        v2 = SelectUserZhongCishuTotal(conn, table_name)
        print("user总次数, {v1}, {v2}".format(v1=v1, v2=v2))

        v1 = SelectUserRobotAchieve(conn, table_name)
        v2 = SelectUserRobotTotal(conn, table_name)
        print("user robot, {v1}, {v2}".format(v1=v1, v2=v2))

        v1 = SelectUserInviteAchieve(conn, table_name)
        v2 = SelectUserInviteTotal(conn, table_name)
        print("user invite, {v1}, {v2}".format(v1=v1, v2=v2))

        for room_id in range(1, 5 + 1):
            v1 = SelectUserRoomAchieve(conn, table_name, room_id)
            v2 = SelectUserRoomTotal(conn, table_name, room_id)
            print("user room, {v1}, {v2}".format(v1=v1, v2=v2))

        conn.close()

if __name__ == "__main__":
    #print(random.randint(10,10000))
    main()
