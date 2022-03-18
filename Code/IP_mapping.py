import pandas as pd

data_features = pd.read_csv('features.features') #featuer file
data_raw = pd.read_csv('ft-v05.2021-03-13.060001-0700.csv') #Raw data #depends on Netflow data

data_features.insert(0,'S_IP', data_raw.loc[:,'srcaddr'])
data_features.insert(1,'D_IP', data_raw.loc[:,'dstaddr'])

#it is better to add this code to the man K-means code file instead of writing it to a csv file to avoid dealing with the data frame indexes
data_features.to_csv('/home/baal7013/CNA/mapped_features')




