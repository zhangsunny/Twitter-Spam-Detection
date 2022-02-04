"""
polluter shape: (22223, 2)
legitimate shape: (19276, 2)
creat table
repeat id: 44
records in user table: 41455
len of followings: 230
"""
import numpy as np
import pandas as pd
import sqlite3


if __name__ == "__main__":
    names = ['UserID', 'followings']
    polluter_path = './honeypot/content_polluters_followings.txt'
    df_p = pd.read_csv(polluter_path, delimiter='\t', header=None, names=names)
    data_p = df_p.values.tolist()
    print('polluter shape:', df_p.shape)

    legit_path = './honeypot/legitimate_users_followings.txt'
    df_l = pd.read_csv(legit_path, delimiter='\t', header=None, names=names)
    data_l = df_l.values.tolist()
    print('legitimate shape:', df_l.shape)

    data = data_p + data_l

    conn = sqlite3.connect('honeypot.db')
    cursor = conn.cursor()

    cursor.execute(
        "select count(*) from sqlite_master where type='table' and name='following'")
    if cursor.fetchone()[0] == 0:
        print('creat table')
        cursor.execute("""create table following (
            id varchar(10) primary key,
            followings text
        )""")
    else:
        print('table exists')
    sql_cmd = 'insert into following values(?, ?)'
    repeat_count = 0
    for i in range(len(data)):
        cursor.execute('select count(*) from following where id=?', [data[i][0]])
        if cursor.fetchone()[0] == 0:
            cursor.execute(sql_cmd, data[i])
        else:
            repeat_count += 1
    print('repeat id:', repeat_count)

    cursor.execute('select count(*) from following')
    print('records in user table:', cursor.fetchone()[0])
    cursor.close()
    conn.commit()
    conn.close()
