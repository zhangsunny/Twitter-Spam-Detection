"""
polluter shape: (22223, 9)
legitimate shape: (19276, 9)
creat table
repeat id: 44
records in user table: 41455
"""
import numpy as np
import pandas as pd
import sqlite3


if __name__ == "__main__":
    names = [
        'UserID', 'CreatDate', 'CollectDate', '#Following',
        '#Follower', '#Tweet', '#ScreenName', '#Profile']
    polluter_path = './honeypot/content_polluters.txt'
    df_p = pd.read_csv(polluter_path, delimiter='\t', header=None, names=names)
    df_p['is_polluter'] = [1] * len(df_p)
    data_p = df_p.values.tolist()
    print('polluter shape:', df_p.shape)

    legit_path = './honeypot/legitimate_users.txt'
    df_l = pd.read_csv(legit_path, delimiter='\t', header=None, names=names)
    df_l['is_polluter'] = [0] * len(df_l)
    data_l = df_l.values.tolist()
    print('legitimate shape:', df_l.shape)

    conn = sqlite3.connect('honeypot.db')
    cursor = conn.cursor()

    cursor.execute(
        "select count(*) from sqlite_master where type='table' and name='user'")
    if cursor.fetchone()[0] == 0:
        print('creat table')
        cursor.execute("""create table user (
            id varchar(10) primary key,
            date_create datetime,
            date_collect datetime,
            num_following int,
            num_follower int,
            num_tweet int,
            len_screen_name int,
            len_profile int,
            is_polluter boolean，
        )""")
    else:
        print('table exists')
    sql_cmd = 'insert into user values(?, ?, ?, ?, ?, ?, ?, ?, ?)'
    cursor.executemany(sql_cmd, data_p)
    repeat_count = 0
    for i in range(len(data_l)):
        cursor.execute('select count(*) from user where id=?', [data_l[i][0]])
        if cursor.fetchone()[0] == 0:
            cursor.execute(sql_cmd, data_l[i])
        else:
            repeat_count += 1
    print('repeat id:', repeat_count)

    cursor.execute('select count(*) from user')
    print('records in user table:', cursor.fetchone()[0])
    cursor.close()
    conn.commit()
    conn.close()
