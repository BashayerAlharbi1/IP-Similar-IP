import pandas as pd



# features data file, without header 
df = pd.read_csv('finalfile1.features',header=None)
#raw data file from netflow .csv
data_raw = pd.read_csv('ft-v05.2021-03-13.060001-0700.csv') 

#add source and destination ips column names to the features file
df.insert(1,'S_IP', data_raw.loc[:,'srcaddr']) 
df.insert(2,'D_IP', data_raw.loc[:,'dstaddr'])

# save headers with the additional ip addresses from the .csv file
df.to_csv("headers_included.csv",header=["Label","S_IP","D_IP","Feature0","Feature1","Feature2","Feature3","Feature4","Feature5","Feature6","Feature7","Feature8","Feature9"])

# list head columns
df.head()

print(df.head())

#list datatype info
df.info()

print(df.info())

#describe data
df.describe()
print(df.describe())



