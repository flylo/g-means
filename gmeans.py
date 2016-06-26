import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import scale

from sklearn import datasets

from scipy.stats import anderson

from pdb import set_trace


class GMeans(object):
	
	"""strictness = how strict should the anderson-darling test for normality be
			0: not at all strict
			4: very strict
	"""

	def __init__(self, min_obs=1, max_depth=10, random_state=None, strictness=4):

		super(GMeans, self).__init__()

		self.max_depth = max_depth
		
		self.min_obs = min_obs

		self.random_state = random_state

		if strictness not in range(5):
			raise ValueError("strictness parameter must be integer from 0 to 4")
		self.strictness = strictness

		self.stopping_criteria = []
		
	def _gaussianCheck(self, vector):
		"""
		check whether a given input vector follows a gaussian distribution
		H0: vector is distributed gaussian
		H1: vector is not distributed gaussian
		"""
		output = anderson(vector)

		if output[0] <= output[1][self.strictness]:
			return True
		else:
			return False
		
	
	def _recursiveClustering(self, data, depth, index):
		"""
		recursively run kmeans with k=2 on your data until a max_depth is reached or we have
			gaussian clusters
		"""
		depth += 1
		if depth == self.max_depth:
			self.data_index[index[:, 0]] = index
			self.stopping_criteria.append('max_depth')
			return
			
		km = MiniBatchKMeans(n_clusters=2, random_state=self.random_state)
		km.fit(data)
		
		centers = km.cluster_centers_
		v = centers[0] - centers[1]
		x_prime = scale(data.dot(v) / (v.dot(v)))
		gaussian = self._gaussianCheck(x_prime)
		
		# print gaussian

		if gaussian == True:
			self.data_index[index[:, 0]] = index
			self.stopping_criteria.append('gaussian')
			return

		labels = set(km.labels_)
		for k in labels:
			current_data = data[km.labels_ == k]

			if current_data.shape[0] <= self.min_obs:
				self.data_index[index[:, 0]] = index
				self.stopping_criteria.append('min_obs')
				return
			

			current_index = index[km.labels_==k]
			current_index[:, 1] = np.random.randint(0,100000000000)
			self._recursiveClustering(data=current_data, depth=depth, index=current_index)

		# set_trace()
	

	def fit(self, data):
		"""
		fit the recursive clustering model to the data
		"""
		self.data = data
		
		data_index = np.array([(i, False) for i in xrange(data.shape[0])])
		self.data_index = data_index

		self._recursiveClustering(data=data, depth=0, index=data_index)

		self.labels_ = self.data_index[:, 1]
		
		
if __name__ == '__main__':
	# iris = datasets.load_iris().data

	iris = datasets.make_blobs(n_samples=10000,
		n_features=2,
		centers=4,
		cluster_std=1.0)[0]

	gmeans = GMeans(random_state=1010,
		strictness=4)
	# # set_trace()
	gmeans.fit(iris)

	plot_data = pd.DataFrame(iris[:, 0:2])
	plot_data.columns = ['x', 'y']
	plot_data['labels_gmeans'] = gmeans.labels_
	# set_trace()
	
	km = MiniBatchKMeans(n_clusters=4)
	km.fit(iris)
	plot_data['labels_km'] = km.labels_
	
	sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_gmeans', fit_reg=False)
	sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_km', fit_reg=False)
	plt.show()
	set_trace()


