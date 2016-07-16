import pandas as pd
import ezcluster

# load the iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/thisisandreeeee/ezcluster/master/iris.csv')
species = iris['species']
iris.drop('species', axis = 1, inplace = True)

# initialize the kmeans class
ezc = ezcluster.Kmeans(iris)

# find the optimal number of clusters
num_of_clusters = ezc.optimal_k(min_k=1, max_k=10, num_iters=1000)
print("Searching for optimal number of clusters from %s to %s" % (min_k, max_k))

# plot the gap statistic plots
# plots are saved to ezcluster_files/ by default
ezc.plot()

# return a model with optimal number of clusters
model = ezc.fit(n_clusters = num_of_clusters)
print("Optimal number of clusters: %s" % num_of_clusters)

# save instance
# model is saved to ezcluster_files/ezc.pkl by default
ezc.save_instance()

# save labeled dataset to csv
# dataframe is saved to ezcluster_files/data.csv by default
ezc.write_csv()
