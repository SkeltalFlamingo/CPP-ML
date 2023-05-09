#-------------------------------------------------------------------------
# AUTHOR: Mason Nash
# FILENAME: clustering.py
# SPECIFICATION: practice training K-Means model. 
# FOR: CS 4210- Assignment #5
# TIME SPENT: 
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df)

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code
highestSilhouette = -1
silhouettes = []
kSizes = list(range(2, 21))
for kSize in kSizes :
     print(f'K Size: {kSize}')
     
     # create the model
     kmeans = KMeans(n_clusters=kSize, random_state=0)
     kmeans.fit(X_training)

     #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     #find which k maximizes the silhouette_coefficient
     #--> add your Python code here
     currentSilhouette = silhouette_score(X_training, kmeans.labels_)
     silhouettes.append(currentSilhouette)
     if currentSilhouette > highestSilhouette :
          highestSilhouette = currentSilhouette
     print(f'highest silhouette socre so far: {highestSilhouette}')

#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here

# create the plot
plt.plot(kSizes, silhouettes)

# add a title and axis labels
plt.title("Silhouette Scores")
plt.xlabel("K Sizes")
plt.ylabel("Silhouette Score")

# display the plot
plt.show()

#reading the test data (clusters) by using Pandas library
#--> add your Python code here
d2 = pd.read_csv('testing_data.csv', sep=',', header=None) #reading the test data by using Pandas library

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here
labels = np.array(d2.values).reshape(1,len(d2.values))[0]

#Calculate and print the Homogeneity of this kmeans clustering
#print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
#--> add your Python code here
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())