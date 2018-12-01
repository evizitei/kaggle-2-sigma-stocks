# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
from datetime import timedelta, datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os
from kaggle.competitions import twosigmanews

class StockStateMachine(object):
    """Takes one days dataframes at a time and can produce the X and Y (when available)
    outputs, persisting state in between days to allow for windowed features over time"""
    def __init__(self):
        self.asset_state = {}

    def empty_record(self):
        # returnsOpenPrevMktres1, returnsOpenPrevMktres10,
        # pos_sent, neg_sent, returnsOpenNextMktres10
        #   - 5 zeros
        return np.zeros(5)

    def preload_news_state(self, ndf):
        for idx, row in ndf.iterrows():
            ac = row['assetCodes']
            codes = ac.replace("{", "").replace("'", "").replace("}", "").split(",")
            for code in codes:
                if code not in self.asset_state:
                    self.asset_state[code] = self.empty_record()
                self.asset_state[code][2] = row['sentimentPositive']
                self.asset_state[code][3] = row['sentimentNegative']

    def process_day_state(self, mdf, ndf, cur_ts, last_ts):
        day_market_data = mdf[mdf['time'] == cur_ts]
        day_news_data = ndf[(ndf['time'] >= last_ts) & (news_df['time'] <= cur_ts)]
        self.preload_news_state(day_news_data)
        rows = []
        for idx, row in day_market_data.iterrows():
            code = row['assetCode']
            if code not in self.asset_state:
                self.asset_state[code] = self.empty_record()
            self.asset_state[code][0] = row['returnsOpenPrevMktres1']
            self.asset_state[code][1] = row['returnsOpenPrevMktres10']
            self.asset_state[code][4] = row['returnsOpenNextMktres10']
            rows.append(self.asset_state[code])
        return np.array(rows)



def produce_training_set(mdf, ndf):
    """
    Iterate through each day window,
    update current state of each stock in universe,
    dump state to raw array
    append to training set
    """
    cur_ts = mdf.iloc[0]['time']
    news_ts = ndf.iloc[0]['time']
    last_ts = mdf.iloc[len(market_df) - 1][0]
    # pre-load state with news information from before the commencement
    # of trading windows
    pre_news = ndf[ndf['time'] < cur_ts]
    machine = StockStateMachine()
    print("Caching News State...")
    machine.preload_news_state(pre_news)
    print("Preloaded State size: ", len(machine.asset_state))
    prev_ts = cur_ts
    training_set = machine.process_day_state(mdf, ndf, cur_ts, prev_ts)
    cur_ts = cur_ts + timedelta(hours=24)
    print("processing day windows...")
    i = 0
    # change iteration strategy to step through each row in order until the TS changes,
    # then snapshot
    while cur_ts <= last_ts:
        i += 1
        new_day_rows = machine.process_day_state(mdf, ndf, cur_ts, prev_ts)
        if len(new_day_rows) > 0:
            training_set = np.concatenate((training_set, new_day_rows))
        prev_ts = cur_ts
        cur_ts = cur_ts + timedelta(hours=24)
        if i % 10 == 0:
            print("...days ", i, "...")
            print("Current Shape: ", training_set.shape, datetime.now())
            print("Current TS", cur_ts)
    print("Dataset Shape: ", training_set.shape)
    return training_set


# Any results you write to the current directory are saved as output.
env = twosigmanews.make_env()
(market_df, news_df) = env.get_training_data()

ts = produce_training_set(market_df, news_df)

cur_ts = market_df.iloc[0]['time']
last_ts = cur_ts
rows = machine.process_day_state(market_df, news_df, cur_ts, (cur_ts + timedelta(hours=24)))
