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

    def process_news_row(self, row):
        ac = row[1]
        codes = ac.replace("{", "").replace("'", "").replace("}", "").split(",")
        for code in codes:
            if code not in self.asset_state:
                self.asset_state[code] = self.empty_record()
            self.asset_state[code][2] = row[2] # pos sentiment
            self.asset_state[code][3] = row[3] # neg sentiment

    def process_market_row(self, row):
        code = row[1]
        if code not in self.asset_state:
            self.asset_state[code] = self.empty_record()
        self.asset_state[code][0] = row[2] # prev 1 market res
        self.asset_state[code][1] = row[3] # prev 10 market res
        self.asset_state[code][4] = row[4] # next 10 market res

    def preload_news_state(self, ndf):
        for idx, row in ndf.iterrows():
            self.process_news_row(row)

    def process_day_state(self, m_rows, n_rows):
        for n_row in n_rows:
            self.process_news_row(n_row)
        day_rows = []
        for m_row in m_rows:
            self.process_market_row(m_row)
            day_rows.append(self.asset_state[m_row[1]])
        return np.array(day_rows)



def produce_training_set(mdf, ndf, already_arrays=False):
    """
    Iterate through each day window,
    update current state of each stock in universe,
    dump state to raw array
    append to training set
    """
    machine = StockStateMachine()
    #print("Caching News State...")
    #machine.preload_news_state(pre_news)
    #print("Preloaded State size: ", len(machine.asset_state))
    #prev_ts = cur_ts
    #training_set = machine.process_day_state(mdf, ndf, cur_ts, prev_ts)
    #cur_ts = cur_ts + timedelta(hours=24)
    print("processing day windows...")
    # get iterators for both dataframes
    training_set = None
    m_i = 0
    m_bound = len(mdf)
    n_i = 0
    n_bound = len(ndf)
    i = 0
    print("Transforming DF to arrays...", datetime.now())
    m_vals = mdf[['time','assetCode','returnsOpenPrevMktres1','returnsOpenPrevMktres10','returnsOpenNextMktres10']].values
    n_vals = ndf[['time','assetCodes','sentimentPositive','sentimentNegative']].values
    print("initializing rows...", datetime.now())
    m_row = m_vals[0]
    n_row = n_vals[0]
    while m_i < m_bound:
        m_rows = []
        cur_ts = m_row[0] # time
        print("Parsing Markets: ", cur_ts, datetime.now())
        while m_row[0] == cur_ts:
            m_rows.append(m_row)
            m_i += 1
            if m_i >= m_bound:
                break
            m_row = m_vals[m_i]
        print("Market Window: ", len(m_rows))
        n_rows = []
        print("Parsing News Til: ", cur_ts, datetime.now())
        while n_row[0] <= cur_ts:
            n_rows.append(n_row)
            n_i += 1
            if n_i >= n_bound:
                break
            n_row = n_vals[n_i]
            if n_i % 10 == 0:
                print("n_i ", n_i, "row time", n_row[0])
        print("News Window: ", len(n_rows))
        print("Processing Window Rows...", datetime.now())
        ts_rows = machine.process_day_state(m_rows, n_rows)
        i += 1
        if training_set is None:
            training_set = ts_rows
        elif len(ts_rows > 0):
            training_set = np.concatenate((training_set, ts_rows))
        print("...days ", i, "...")
        print("Current Shape: ", training_set.shape, datetime.now())
        print("Current TS", cur_ts)
    print("Dataset Shape: ", training_set.shape)
    return training_set


# Any results you write to the current directory are saved as output.
env = twosigmanews.make_env()
(market_df, news_df) = env.get_training_data()

ts = produce_training_set(market_df, news_df)
