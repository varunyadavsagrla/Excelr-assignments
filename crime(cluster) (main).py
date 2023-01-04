#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import warnings 
warnings.filterwarnings('ignore')


# In[60]:


data=pd.read_csv('crime_data.csv')
data


# In[61]:


data.info()


# In[62]:


crime=data.drop("Unnamed: 0",axis=1)
crime


# Normalization

# In[63]:


def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[64]:


df_norm = norm_func(crime.iloc[:,:])
df_norm 


# #KMEANS Clustering

# In[65]:


fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()  


# In[66]:


clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(df_norm)  


# In[67]:


y_kmeans
#clf.cluster_centers_
clf.labels_ 


# In[68]:


y_kmeans 


# In[69]:


clf.cluster_centers_ 


# In[70]:


clf.inertia_


# In[71]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
crime['clust']=md # creating a  new column and assigning it to new column 
crime


# In[72]:


crime.groupby(crime.clust).mean() 


# In[73]:


WCSS


# KMeans visualization

# In[74]:


plt.figure(figsize=(15,8))
sn.scatterplot(crime['clust'],data['Unnamed: 0'],c=clf.labels_,s=300,marker='*')
plt.show();


# # DBSCAN Clustering

# In[75]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


# In[76]:


crime


# In[77]:


array=crime.values
array


# In[78]:


stscaler = StandardScaler().fit(array)
X = stscaler.transform(array) 
X  


# In[79]:


dbscan = DBSCAN(eps=1.25, min_samples=5)
dbscan.fit(X)


# In[80]:


#Noisy samples are given the label -1.
dbscan.labels_          


# In[81]:


c=pd.DataFrame(dbscan.labels_,columns=['cluster'])   


# In[82]:


c
pd.set_option("display.max_rows", None)  


# In[83]:


c


# In[84]:


df = pd.concat([data,c],axis=1)  
df     


# In[85]:


d1=dbscan.labels_
d1


# In[86]:


import sklearn
sklearn.metrics.silhouette_score(X, d1) 


# In[87]:


from sklearn.cluster import KMeans
clf = KMeans(n_clusters=5)
y_kmeans = clf.fit_predict(X)


# In[88]:


y_kmeans


# In[89]:


cl1=pd.DataFrame(y_kmeans,columns=['Kcluster']) 
cl1


# In[90]:


df1 = pd.concat([df,cl1],axis=1) 
df1 


# Silhoutte_score  

# In[91]:


sklearn.metrics.silhouette_score(X, y_kmeans)


# DBSCAN Visualization

# In[92]:


df.plot(x="Unnamed: 0",y ="cluster",c=dbscan.labels_ ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using DBScan')      


# In[93]:


plt.figure(figsize=(15,8))
sn.scatterplot(df1['Kcluster'],df1['Unnamed: 0'],c=clf.labels_,s=300,marker='*')
plt.show();


# In[94]:


df1.plot(x="Unnamed: 0",y ="Kcluster",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans') 


# #HIERARCHAICAL Clustering

# In[95]:


data


# In[96]:


crime


# ####Standard Scaler

# In[97]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
crime_subset = pd.DataFrame(scaler.fit_transform(crime.iloc[:,1:7]))
crime_subset  


# ###Dendrogram

# In[98]:


from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch # for creating dendrogram 
p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels
)
plt.show()    


# In[99]:


p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels
)
plt.show()    


# In[100]:


p = np.array(df_norm) # converting into numpy array format 
z = linkage(df_norm, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels
)
plt.show()    


# In[101]:


p = np.array(crime_subset) # converting into numpy array format 
z = linkage(crime_subset, method="complete",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
    z,
    #leaf_rotation=6.,  # rotates the x axis labels
    #leaf_font_size=15.,  # font size for the x axis labels
)
plt.show()


# In[102]:


from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=5, linkage='complete',affinity = "euclidean").fit(df_norm) 

cluster_labels=pd.Series(h_complete.labels_)
cluster_labels
crime['clust']=cluster_labels # creating a  new column and assigning it to new column 
crime   


# In[103]:


crime.iloc[:,1:].groupby(crime.clust).mean()


# In[104]:


data = crime[(crime.clust==0)]
data  


# In[105]:


data = crime[(crime.clust==1)]
data  


# In[106]:


data = crime[(crime.clust==2)]
data  


# In[107]:


data = crime[(crime.clust==3)]
data  


# In[108]:


data = crime[(crime.clust==4)]
data  


# ###Inference

# In Hierarchical cluster, Complete method is suitable for clustering the crime data.   
