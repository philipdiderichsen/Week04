
"""
Exercise 4.1:

Implement the DBSCAN clustering algorithm to work with Jaccard-distance as its metric. It should be able to handle sparse data.

We suggest that you implement the DBSCAN clustering algorithm as described in the Wikipedia article, linked to above. The algorithm should use Jaccard-distance (1 minus Jaccard index) when measuring distance between points. In this exercise two points that are exactly a distance epsilon apart are taken to be in the same cluster, thus greater-than-or-equal dist(a, b) <= epsilon and not dist(a, b) < epsilon should be used.

Your method should support a large number (~100,000) of sparse points in very high-dimensional space (~100,000 dimensions). This means that the points – even though they lie in very high dimensional space – only have few non-zero coordinates.

Debug and test your algorithm with the following:

Use this data. 
https://www.dropbox.com/s/jblvikb87j69rsp/test_files.zip?dl=0
The files are “pickled”, and must first be loaded with pickle. Parse the loaded data to a compressed sparse row matrix using csr_matrix, where each row corresponds to a point and each column to a dimension.
For the five input files, use the following parameters
File 1: eps=0.4 and M=2
File 2: eps=0.3 and M=2
File 3: eps=0.15 and M=2
File 4: eps=0.15 and M=2
File 5: eps=0.15 and M=2
You should get this many clusters when running your algorithm on the five test files (also counting the special cluster of points not assigned to a cluster): 4, 6, 9, 394, 1692
Remember to describe how you process the data in order for the algorithm to work with very high dimensional data.

You should also document how many points is contained in the largest cluster for each of the 5 test files. The correct answers for this will be released for giving peer feedback.

It is much more important to have a solution which can only solve the first few input files correctly than a fancy solution solving none of them. There will be partial credit for the exercise even if you can only make a solution for the smallest input file.
"""

import numpy as np
import pickle

#---------------------------------------------------------------------------------
# 10 points in 10-dim space
#---------------------------------------------------------------------------------

mat10 = pickle.load(open('data/data_10points_10dims.dat', 'rb'), encoding='latin1') 
print('\nmat10\n', type(mat10))

# Represent data as 10 x 10 numpy array
mat10dense = np.asarray(mat10.todense())
print(mat10dense)

def jaccard_dist(arr1, arr2): 
  """
  Calculate the Jaccard distance between two arrays/vectors. 
  :param arr1: Numpy array
  :param arr2: Numpy array
  :returns: Float: Jaccard distance
  """

  # Intersection: Sum of element-wise AND comparison (True is treated as 1)
  intersection = np.sum(np.logical_and(arr1, arr2))

  # Union: Sum of element-wise OR comparison (True is treated as 1)
  union = np.sum(np.logical_or(arr1, arr2))

  jaccard_dist = 1 - intersection / union

  return jaccard_dist

print(jaccard_dist(mat10dense[0], mat10dense[1]))
print(jaccard_dist(mat10dense[0], mat10dense[2]))




mat100 = pickle.load(open('data/data_100points_100dims.dat', 'rb'), encoding='latin1') 
mat1000 = pickle.load(open('data/data_1000points_1000dims.dat', 'rb'), encoding='latin1') 
mat10000 = pickle.load(open('data/data_10000points_10000dims.dat', 'rb'), encoding='latin1') 
mat100000 = pickle.load(open('data/data_100000points_100000dims.dat', 'rb'), encoding='latin1') 

mat100000dense = np.asarray(mat100000.todense())
print(jaccard_dist(mat100000dense[0], mat100000dense[1]))


#print('\nmat100\n', type(mat100))
#print(mat100)

#print('\nmat1000\n', type(mat1000))
#print(mat1000)

#print('\nmat10000\n', type(mat10000))
#print(mat10000)

#print('\nmat100000\n', type(mat100000))
#print(mat100000)
