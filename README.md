# ezcluster
Ezcluster implements simple ways of finding the optimal number of clusters for various unsupervised learning methods, such as [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). Determining the best number of clusters is a tricky task, which ezcluster aims to simplify through use of criterions such as the gap statistic, as proposed in the paper from [Data Science Lab](https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/).

## Example
```python
import pandas as pd
import ezcluster

# load the iris dataset
import pandas as pd
import ezcluster

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
```
