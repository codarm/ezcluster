import pandas as pd
import ezcluster

# load the iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/thisisandreeeee/ezcluster/master/iris.csv')
species = iris['species']
iris.drop('species', axis = 1, inplace = True)

# initialize the kmeans class
ezc = ezcluster.Kmeans(iris)

# find the optimal number of clusters
num_of_clusters = ezc.optimal_k(min_k=1, max_k=10, num_iters=100)

# plot the gap statistic plots
ezc.plot()

# return a model with optimal number of clusters
model = ezc.fit(n_clusters = num_of_clusters)
print(num_of_clusters)

# save instance
ezc.save_model()

# save labeled dataset to csv
ezc.write_csv()
