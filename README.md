# G-Means
Implementation of the [G-Means algorithm](http://papers.nips.cc/paper/2526-learning-the-k-in-k-means.pdf) for learning k in a k-means clustering.  Paper published in NIPS 2003

## Parameters
*  min_obs:  the minimum number of observations that a cluster can have
*  max_depth:  the maximum number of times that a cluster can be split before giving up (just set this to be high, e.g. 200 or so)
*  random_state:  the random seed that sklearn.MiniBatchKMeans() uses
*  strictness:  how strict should the Anderson-Darling test for normality be - 0 is least strict, 4 is most strict (best to be either 3 or 4, since the test is run so many times)

## Usage
```python
gmeans = GMeans(min_obs=100,
	max_depth=500,
	random_state=1010,
	strictness=3)
gmeans.fit(iris)
gmeans.labels_
```
