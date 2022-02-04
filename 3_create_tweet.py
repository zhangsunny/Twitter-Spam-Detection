"""
polluter shape: (2333691, 4)
legitimate shape: (3246377, 4)
table exists
repeat id_tweet: 711
"""
import numpy as np
import pandas as pd
import sqlite3
import time


if __name__ == "__main__":
    names = ['UserID', 'TweetID', 'Content', 'CreateDate']
    polluter_path = './honeypot/content_polluters_tweets.txt'
    df_p = pd.read_csv(polluter_path, delimiter='\t', header=None, names=names)
    data_p = df_p.values.tolist()
    print('polluter shape:', df_p.shape)

    legit_path = './honeypot/legitimate_users_tweets.txt'
    df_l = pd.read_csv(legit_path, delimiter='\t', header=None, names=names)
    data_l = df_l.values.tolist()
    print('legitimate shape:', df_l.shape)

    data = data_p + data_l

    conn = sqlite3.connect('honeypot.db')
    cursor = conn.cursor()

    cursor.execute(
        "select count(*) from sqlite_master where type='table' and name='tweet'")
    if cursor.fetchone()[0] == 0:
        print('creat table')
        cursor.execute("""create table tweet (
            id varchar(10),
            id_tweet varchar(20) primary key,
            content text,
            date_tweet datetime
        )""")
    else:
        print('table exists')
    sql_cmd = 'insert into tweet values(?, ?, ?, ?)'
    # cursor.executemany(sql_cmd, data)
    repeat_count = 0
    # cursor.executemany(sql_cmd, data_p)
    tic = time.time()
    for i in range(len(data)):
        cursor.execute('select count(*) from tweet where id_tweet=?', [data[i][1]])
        if cursor.fetchone()[0] == 0:
            cursor.execute(sql_cmd, data[i])
        else:
            repeat_count += 1
    toc = time.time()
    print('repeat id_tweet:', repeat_count)
    print('time cost:', (toc-tic))
    cursor.close()
    conn.commit()
    conn.close()
