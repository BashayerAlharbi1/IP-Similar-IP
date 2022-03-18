import pandas as pd

data_features = pd.read_csv('features.features') #featuer file
data_raw = pd.read_csv('ft-v05.2021-03-13.060001-0700.csv') #Raw data #depends on Netflow file

#it is better to add column names to the features file
data_features.insert(1,'S_IP', data_raw.loc[:,'srcaddr']) #first columns is zeros
data_features.insert(2,'D_IP', data_raw.loc[:,'dstaddr'])

#it is better to add this code to the  K-means code file instead of writing it to a csv file to avoid dealing with the data frame indexes
data_features.to_csv('/home/baal7013/CNA/mapped_features')

#Note: we dont not need the first three columns when clustering becuase:
#columns 0 has zeros
#columns 1 has source Ips
#columns 1 has dest Ips


