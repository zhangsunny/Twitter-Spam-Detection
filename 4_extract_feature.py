"""
all the denominators are added with an integer to avoid the invalid division
which also conforms to the Dirichlet uncertainty assumption
Read the data from the database and construct the feature data set
"""
import numpy as np
import pandas as pd
import sqlite3
import time
from datetime import datetime

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
UNION_DATE = datetime.strptime('2019-01-01 00:00:00', DATE_FORMAT)


def parse_date(cur):
    sql_cmd = 'select date_create from user'
    dates = cur.execute(sql_cmd).fetchall()
    dates = list(map(lambda x: datetime.strptime(x[0], DATE_FORMAT), dates))
    deltas = list(map(lambda x: getattr(UNION_DATE-x, 'days'), dates))
    return deltas


def parse_content(cur, uids):
    sql_cmd = """
        select count(*) from tweet where id=? and content like ?
    """
    ret = []
    count = 0
    for uid in uids:
        print('{}/{}'.format(count, len(uids)))
        count += 1
        num = cur.execute('select count(*) from tweet where id=?', (uid,)).fetchone()[0]
        ratio_hashtag = cur.execute(sql_cmd, (uid, "%#%")).fetchone()[0] / (num + 1)
        ratio_url = cur.execute(sql_cmd, (uid, "%http:%")).fetchone()[0] / (num + 1)
        ratio_mention = cur.execute(sql_cmd, (uid, "%@%")).fetchone()[0] / (num + 1)
        ret.append([ratio_hashtag, ratio_url, ratio_mention])
    return np.array(ret)


def parse_freq(cur):
    sql_cmd = 'select date_create, date_collect, num_tweet from user'
    result = cur.execute(sql_cmd).fetchall()
    date_create = list(map(lambda x: datetime.strptime(x[0], DATE_FORMAT), result))
    date_collect = list(map(lambda x: datetime.strptime(x[1], DATE_FORMAT), result))
    deltas = list(map(lambda x: getattr(x[1]-x[0], 'days'), zip(date_create, date_collect)))
    num_tweet = list(map(lambda x: x[2], result))
    deltas = np.array(deltas)
    num_tweet = np.array(num_tweet)
    return num_tweet / (deltas + 1)


if __name__ == "__main__":
    conn = sqlite3.connect('honeypot.db')
    cursor = conn.cursor()
    cursor.execute(""" select
        id, num_following, num_follower, num_tweet,
        len_screen_name, len_profile, is_polluter
        from user""")
    users = np.array(cursor.fetchall())
    # user based features
    df = pd.DataFrame(data=users, columns=[
        'id', 'num_following', 'num_follower', 'num_tweet',
        'len_screen_name', 'len_profile', 'is_polluter'], dtype=int)
    df['age'] = parse_date(cursor)
    df['reputation'] = df['num_follower'] / (df['num_follower'] + df['num_following'] + 1)
    df['ff_ratio'] = df['num_following'] / (df['num_follower'] + 1)
    df['freq_tweet'] = parse_freq(cursor)

    # tic = time.time()
    # uids = df['id']
    # ret = parse_content(cursor, uids)
    # df['ratio_hashtag'] = ret[:, 0]
    # df['ratio_url'] = ret[:, 1]
    # df['ratio_mention'] = ret[:, 2]
    # print(df.tail(3))
    # toc = time.time()
    # print('time cost:', (toc-tic))

    print(df.tail(5))
    df.to_csv('./data.csv', index=False)
    cursor.close()
    conn.close()
