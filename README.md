# Ezcluster: Evaluating the optimal number of clusters for KMeans clustering using the gap statistic
[K-Means clustering](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) provides us with interesting ways of exploring our dataset, by trying to separate the data points into K different clusters. However, determining the optimal number of clusters can be a tricky task. Borrowing from the concepts outlined in [this paper](https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/), ezcluster will make it easy to find an optimal K by using the gap statistic.

Installation through the Python Package index:
```
pip install ezcluster
```

## Implementation
The optimal K is the smallest for which the quantity plotted in blue bars becomes positive.

![alt text](https://github.com/thisisandreeeee/ezcluster/blob/master/gaps_with_error.png "Optimal K")

## Documentation
`ezcluster.KMeans`
Initializes the class with an input dataframe, which is preprocessed in preparation for K-Means clustering
- Input parameters:
    - df (DataFrame): pandas dataframe
    - categorical_cols (list of strings): columns to be one hot encoded
    - id_col (string): name of id column

`KMeans.optimal_k`
Searches for the optimal K within a supplied range
- Input parameters:
    - min_k (int): minimum k to start search
    - max_k (int): maximum k to end search
    - num_iters (int): number of times to run K-Means; the more the better because law of large numbers averages out the results, but takes longer to run

`KMeans.plot`
Generates the `gap_statistic` and `gaps_with_error` plots, and saves them by default to the ezcluster_files/ directory unless otherwise specified.

`KMeans.fit`
Returns a KMeans object initialized with the optimal number of clusters supplied to it.

`KMeans.save_model`
Saves the ezcluster instance as a .pkl file as ezcluster_files/ezc.pkl by default unless otherwise specified.

`KMeans.load_model`
Loads the previous saved ezcluster instance from file.

`KMeans.write_csv`
Saves labeled dataframe to csv file, in ezcluster_files/ unless otherwise specified.

## Example
Import your packages and load the iris dataset in a pandas dataframe.
```python
# load the iris dataset
import pandas as pd
import ezcluster

iris = pd.read_csv('https://raw.githubusercontent.com/thisisandreeeee/ezcluster/master/iris.csv')
species = iris['species']
iris.drop('species', axis = 1, inplace = True)
```

```python
# initialize the kmeans class with a pandas dataframe, and indicate the categorical or id columns
ezc = ezcluster.Kmeans(iris, categorical_cols = None, id_col = None)

# find the optimal number of clusters by indicating the range of K to try
num_of_clusters = ezc.optimal_k(min_k=1, max_k=10, num_iters=100)

# plot the gap statistic plots
ezc.plot()

# return a model with optimal number of clusters
model = ezc.fit(n_clusters = num_of_clusters)

# save instance
ezc.save_instance()

# save labeled dataset to csv
ezc.write_csv()
```
