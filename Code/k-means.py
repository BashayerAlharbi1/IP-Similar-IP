import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()# Bashayer noted do we need this??? not sure.
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import shap
import argparse

#---------------------------------------------------------------------------------------------------------#
#################################### Preparing the Data for Analysis ######################################
#---------------------------------------------------------------------------------------------------------#
def dataNormalization(data):
    
    #Normalizing the data
    data_value = data.copy()
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled = min_max_scaler.fit_transform(data_value)
    data_df_scaled = pd.DataFrame(data_scaled, columns=data.columns)

    #Drop the first column that only contains 0s
    data_df_scaled.drop(columns=data_df_scaled.columns[0],axis=1,inplace=True)
    return clustering(data_df_scaled)

#---------------------------------------------------------------------------------------------------------#
############################################### Clustering ################################################
#---------------------------------------------------------------------------------------------------------#
def clustering(data_df_scaled):

    ############## Selecting the Optimal Number of k ##############
    #The elbow method to select the optimal number of k:
    wcss=[]
    for i in range(1,10): #tries to cluster the data using different value of k, starting from k=1 to k=10.
        kmeans = KMeans(n_clusters =i, init='k-means++')
        kmeans.fit(data_df_scaled)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    number_clusters = range(1,10)
    plt.figure(figsize=(10, 5))
    plt.plot(number_clusters,wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.savefig('Elbow.png') # to show the plot 
    plt.show() 
    plt.clf() #To clear the current figure

    ######################## Kmeans Cluster #######################
    #Clustering: Determines how many clusters are used
    #For now we choose the optimal number produced by the elbow method
    kmeans = KMeans(n_clusters =6 , init='k-means++')
    kmeans.fit(data_df_scaled)
    #Clustering result 
    identified_clusters = kmeans.fit_predict(data_df_scaled)
    data_with_clusters = data_df_scaled.copy()
    data_with_clusters['Clusters'] = identified_clusters
    frame = pd.DataFrame(data_with_clusters)
    #prints the size of each cluster  
    print (frame['Clusters'].value_counts()) 

    ############## Principal component analysis (PCA) #############
    pca_num_components = 2 # to determine the number of dimensions, for now, we choose 2 dimensions. 
    reduced_data = PCA(n_components=pca_num_components).fit_transform(data_with_clusters)
    results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
    sns.scatterplot(x="pca1", y="pca2", hue=data_with_clusters['Clusters'], data=results)
    plt.title('K-means Clustering with 2 dimensions')
    plt.savefig('pca_clusters.png')
    plt.clf()
    centroids = kmeans.cluster_centers_
    centroids = pd.DataFrame(centroids, columns=['DpktsSA','DpktsSV','DoctetsSA','DoctetsSV',
                                                 'DpktsDA','DpktsDV','DoctetsDA','DoctetsDV','AveTimeV','AveTimeA'])
    centroids.index = np.arange(1, len(centroids)+1) # Start the index from 1
    reduced_data = PCA(n_components=pca_num_components).fit_transform(centroids)
    results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])
    sns.scatterplot(x="pca1", y="pca2", hue=['Centroid0','Centroid1','Centroid2','Centroid3', 'Centroid4', 'Centroid5'], data=results)
    plt.savefig('pca_centroids.png')
    plt.clf()

    #another way to plot the centroids w/o using pca
    #plt.scatter(centroids['DpktsSA'], centroids['DpktsDA'], marker="x", color='r')
    #plt.legend()
    #plt.savefig('centroids.png')
    
    return data_with_clusters 

#---------------------------------------------------------------------------------------------------------#
#################################### Mapping IPs along with Features ######################################
#---------------------------------------------------------------------------------------------------------#
def mappingIPs(raw_data, data_with_clusters):
    
    #add the IP addresses along with features and clusters
    data_with_clusters.insert(0,'S_IP', raw_data.loc[:,'srcaddr']) #first column is zeros
    data_with_clusters.insert(1,'D_IP', raw_data.loc[:,'dstaddr'])
    print(data_with_clusters.loc[data_with_clusters['Clusters'] == 1]) #print all the data entries that belongs to cluster 1
    print(data_with_clusters.loc[data_with_clusters['Clusters'] == 0]) #print all the data entries that belongs to cluster 0
    data_with_clusters.to_csv('Final_Mapped_Features') #this file contains the IP addresses along with their classes (e.g, class 0 or class 1)
    
#---------------------------------------------------------------------------------------------------------#
############################## Using SHAP to explain the clustering model #################################
#---------------------------------------------------------------------------------------------------------#
def SHAPexplainer(data_with_clusters):

    """
        We used the cluserts' numbers as lables for our dataset
    """  
    y = data_with_clusters.loc[:,['Clusters']] #labels
    x = data_with_clusters.loc[:, ['DpktsSA','DpktsSV','DoctetsSA','DoctetsSV',
                                   'DpktsDA','DpktsDV','DoctetsDA','DoctetsDV',
                                   'AveTimeV','AveTimeA']] #the rest of the data without the lables
    clf = RandomForestClassifier()
    clf.fit(x, y)
    explainer = shap.TreeExplainer(clf)
    shap_values = [] #To store the Shapley values

    #Generating the summaty plot
    shap_values = explainer.shap_values(x) #applying the explainer on the test_x to check what featuer have more impact
    shap.summary_plot(shap_values, x) #present a summary plot of all of the features ordered based on importance
    plt.show()

def main():

    #Process command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--FeaturesFile', type=str, required=True,
                        help="The csv file with the extracted features.")   
    parser.add_argument('--RawDatafile', type=str, required=True,
                        help="The csv raw data file")
    parser.add_argument('--SHAP', action='store_true',
                        help="This option to explain the clustered data; it is optional")
    wFLAGS = parser.parse_args()
    
    try:
        
        featuers_data = pd.read_csv(FLAGS.FeaturesFile)
        raw_data = pd.read_csv(FLAGS.RawDatafile) #Raw data #depends on Netflow file
        clusteredData = dataNormalization(featuers_data) #Returned dataframe to be used in SHAPexplainer() and mappingIPs()

        mappingIPs(raw_data, clusteredData) #To map the extracted features with the the IPs from the raw data before extraction
        
    except FileNotFoundError:
        print("File Doesn't exist.")
    except:
        print("There was an error with on of the files.")

    if FLAGS.SHAP: #If SHAP option is selcted
        SHAPexplainer(clusteredData) #To explain which features contribute more to the clustering process using Shapley values
        
main()  

