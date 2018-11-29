# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

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

    def empty_record():
        # returnsOpenPrevMktres1, returnsOpenPrevMktres10,
        # pos_sent, neg_sent, returnsOpenNextMktres10
        #   - 5 zeros
        return np.zeros(5)

    def preload_news_state(self, ndf):
        for row in pre_news.iterrows():
            ac = row['assetCodes']
            codes = ac.replace("{", "").replace("'", "").replace("}", "").split(",")
            for code in codes:
                if code not in self.asset_state:
                    self.asset_state[code] = self.empty_record()
                self.asset_state[code][2] = row['sentimentPositive']
                self.asset_state[code][3] = row['sentimentNegative']
            print(row.assetCodes.split(","))


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


# Any results you write to the current directory are saved as output.
env = twosigmanews.make_env()
(market_df, news_df) = env.get_training_data()

produce_training_set(market_df, news_df)
