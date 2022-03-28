import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mping
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
#:unix_secs,unix_nsecs,sysuptime,exaddr,dpkts,doctets,first,last,engine_type,engine_id,srcaddr,dstaddr,nexthop,input,output,srcport,dstport,prot,tos,tcp_flags,src_mask,dst_mask,src_as,dst_as


#read .csv with headers included
df = pd.read_csv("headers_included.csv")#,encoding="ISO-8859-1")
df = pd.DataFrame(df, columns = ['S_IP','Feature1','Feature2','Feature3'])


# cat codes
df['S_IP'] = pd.Categorical(df["S_IP"])
df["S_IP"] = df["S_IP"].cat.codes
df['Feature1'] = pd.Categorical(df["Feature1"])
df["Feature1"] = df["Feature1"].cat.codes
df['Feature2'] = pd.Categorical(df["Feature2"])
df["Feature2"] = df["Feature2"].cat.codes
df['Feature3'] = pd.Categorical(df["Feature3"])
df["Feature3"] = df["Feature3"].cat.codes


x = df.iloc[:, [0, 1,2,3]].values
df.head()
#print(df.head())

scaler = MinMaxScaler()
norm_df = df.copy()
def minmaxscaler(x):
    for Protocol, Time in x.iteritems():
        x[Protocol] = scaler.fit_transform(np.array(Time).reshape(-1, 1))
    
minmaxscaler(norm_df)
norm_df.head()
print (norm_df.head())



k = list(range(1,10))
sum_of_squared_distances = []
for i in k:
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(norm_df)
    sum_of_squared_distances.append(kmeans.inertia_)
plt.figure(figsize=(10, 5))
plt.plot(k, sum_of_squared_distances, 'go--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within Cluster Sum of squares')
plt.title('Elbow Curve to find optimum K')
plt.savefig('kmeans_f.png')
#img=Image.open('kmeans_f.png')
#img.show()

# Instantiating
kmeans5 = KMeans(n_clusters = 5)

# Training the model
kmeans5.fit(norm_df)

# Predicting
y_pred = kmeans5.fit_predict(norm_df)
print(y_pred)

# Storing the y_pred values in a new column
df['Cluster'] = y_pred+1 #to start the cluster number from 1

centroids = kmeans5.cluster_centers_
centroids = pd.DataFrame(centroids, columns=['S_IP','Feature1','Feature2','Features'])
centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1
centroids
print(centroids)

#import seaborn as sns
plt.figure(figsize=(12,6))
sns.set_palette("pastel")
sns.scatterplot(x=df['S_IP'], y = df['Feature2'], hue=df['Cluster'], palette='bright')
plt.savefig('kmeansscpltf2.png')
#img=Image.open('kmeansscpltf.png')
#img.show()

