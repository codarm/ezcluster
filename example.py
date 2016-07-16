import pandas as pd
import ezcluster

iris = pd.read_csv('iris.csv') # change to github link
species = iris['species']
iris.drop('species', axis = 1, inplace = True)

ezc = ezcluster.Kmeans(iris)
ezc.fit()
ezc.write_csv()
