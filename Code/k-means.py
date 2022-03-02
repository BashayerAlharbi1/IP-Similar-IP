#This is an initial code that we might use for the K-means method, we still need to add some changes.
#Libraries that we need.
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

#First we need to load the csv data file 
data = pd.read_csv('ft-v05.2021-03-13.060001-0700.csv'

#OPTHIONAL: Ploting the data (we need to choose which columns we want to plot) 
plt.scatter(data['prot'],data['dpkts'])
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()
)

#Selcting the features
x = data.iloc[:,1:3] # 1t for rows and second for columns

#Clustring (we should determine how many clusters we want to create)
#For now we choose 4)
kmeans = KMeans(4)
means.fit(x)

#Clustring result
identified_clusters = kmeans.fit_predict(x)
identified_clusters

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = identified_clusters 
plt.scatter(data_with_clusters['prot'],data_with_clusters['dpkts'],c=data_with_clusters['Clusters'],cmap='rainbow')

