import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import sys
import logging
import argparse

from os.path import abspath
sys.path.insert(0, abspath('..'))


from sklearn import metrics
from sklearn.cluster import KMeans
from spherecluster import SphericalKMeans, VonMisesFisherMixture
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import numpy as np
from scipy.sparse import csr_matrix
from tabulate import tabulate

from data.data_loader import load_data
from evaluate import Evaluate
from utils import SphericalKmeans, SphericalKmeansPlus

import umap


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='biomedical',
						choices=['stackoverflow', 'biomedical', 'searchsnippets'])
	parser.add_argument('--dim_red', default='UMAP', choices=['LSA', 'ACP', 'UMAP', 'TSNE'])
	parser.add_argument('--word_emb', default='HuggingFace', choices=['HuggingFace', 'TF-IDF', 'Jose'])
	args = parser.parse_args()

	if args.dataset == 'searchsnippets':
		dataset_path = 'datasets/SearchSnippets'
	elif args.dataset == 'stackoverflow':
		dataset_path = 'datasets/stackoverflow'
	elif args.dataset == 'biomedical':
		dataset_path = 'datasets/Biomedical'
	else:
		raise ValueError("Invalid dataset")
	

	#plot = SpacePlot()
	eval = Evaluate()

	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
	
	###############################################################################
	# Data loading

	x, y = load_data(dataset=dataset_path, word_emb=args.word_emb, transform=None, scaler=None, norm=None)
	n_clusters = len(np.unique(y))

	print("%d documents" %  x.shape[0])
	print("%d categories" % n_clusters)
	print()
	print('Original shape:', x.shape, y.shape)

	###############################################################################
	# UMAP for dimensionality reduction (and finding dense vectors)
	if args.dim_red == 'UMAP':
		print("Performing dimensionality reduction using UMAP")
		n_components = 100
		u = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, metric='cosine')
		normalizer = Normalizer(copy=False)
		u = make_pipeline(u, normalizer)
		x = u.fit_transform(x)

		print()
	###############################################################################
	# LSA for dimensionality reduction (and finding dense vectors)
	if args.dim_red == 'LSA':
		print("Performing dimensionality reduction using LSA")
		n_components = 150
		svd = TruncatedSVD(n_components)
		normalizer = Normalizer(copy=False)
		lsa = make_pipeline(svd, normalizer)
		x = lsa.fit_transform(x)

		explained_variance = svd.explained_variance_ratio_.sum()
		print("Explained variance of the SVD step: {}%".format(
			int(explained_variance * 100)))

		print()
	
	###############################################################################
	# ACP for dimensionality reduction (and finding dense vectors)
	if args.dim_red == 'ACP':
		print("Performing dimensionality reduction using ACP")
		n_components = 100
		acp = PCA(n_components)
		normalizer = Normalizer(copy=False)
		acp = make_pipeline(acp, normalizer)
		x = acp.fit_transform(x)

		print()

		
	print('Reduced shape:', x.shape)


	# table for results display
	table = []



	###############################################################################
	# K-Means++ clustering
	km = KMeans(n_clusters=n_clusters, init='random', n_init=20, random_state=1000)
	
	print("Clustering with %s" % km)
	km.fit(x)
	print()
	
	print("Accuracy: %.3f" % eval.accuracy(y, km.labels_))
	print("Normalized Mutual Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, km.labels_))
	print("Adjusted Rand-Index: %.3f" 
	   % metrics.adjusted_rand_score(y, km.labels_))
	print("Adjusted Mututal Information: %.3f" 
	   % metrics.adjusted_mutual_info_score(y, km.labels_))
	print("Normalized Mututal Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, km.labels_))
	print("Silhouette Coefficient (euclidean): %0.3f" 
	   % metrics.silhouette_score(x, km.labels_, metric='euclidean'))
	print("Silhouette Coefficient (cosine): %0.3f" 
	   % metrics.silhouette_score(x, km.labels_, metric='cosine'))
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, km.labels_))
	print("Completeness: %0.3f" % metrics.completeness_score(y, km.labels_))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, km.labels_))

	print()

	table.append([
		'k-means',
		eval.accuracy(y, km.labels_),
		metrics.normalized_mutual_info_score(y, km.labels_),
		metrics.adjusted_rand_score(y, km.labels_),
		metrics.adjusted_mutual_info_score(y, km.labels_),
		metrics.homogeneity_score(y, km.labels_),
		metrics.completeness_score(y, km.labels_),
		metrics.v_measure_score(y, km.labels_),
		metrics.silhouette_score(x, km.labels_, metric='cosine'),
		metrics.silhouette_score(x, km.labels_, metric='euclidean')])
	

	###############################################################################
	# K-Means++ clustering
	kmp = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=20, random_state=1000)
	
	print("Clustering with %s++" % kmp)
	kmp.fit(x)
	print()
	
	print("Accuracy: %.3f" % eval.accuracy(y, kmp.labels_))
	print("Normalized Mutual Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, kmp.labels_))
	print("Adjusted Rand-Index: %.3f" 
	   % metrics.adjusted_rand_score(y, kmp.labels_))
	print("Adjusted Mututal Information: %.3f" 
	   % metrics.adjusted_mutual_info_score(y, kmp.labels_))
	print("Normalized Mututal Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, kmp.labels_))
	print("Silhouette Coefficient (euclidean): %0.3f" 
	   % metrics.silhouette_score(x, kmp.labels_, metric='euclidean'))
	print("Silhouette Coefficient (cosine): %0.3f" 
	   % metrics.silhouette_score(x, kmp.labels_, metric='cosine'))
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, kmp.labels_))
	print("Completeness: %0.3f" % metrics.completeness_score(y, kmp.labels_))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, kmp.labels_))

	print()

	table.append([
		'k-means++',
		eval.accuracy(y, kmp.labels_),
		metrics.normalized_mutual_info_score(y, kmp.labels_),
		metrics.adjusted_rand_score(y, kmp.labels_),
		metrics.adjusted_mutual_info_score(y, kmp.labels_),
		metrics.homogeneity_score(y, kmp.labels_),
		metrics.completeness_score(y, kmp.labels_),
		metrics.v_measure_score(y, kmp.labels_),
		metrics.silhouette_score(x, kmp.labels_, metric='cosine'),
		metrics.silhouette_score(x, kmp.labels_, metric='euclidean')])
	

	###############################################################################
	# Spherical K-Means clustering
	skm = SphericalKmeans(n_clusters=n_clusters, max_iter=300, n_init=20, weighting=True, random_state=1000)
	
	print("Clustering with %s" % skm)
	skm.fit(x)
	print()
	
	print("Accuracy: %.3f" % eval.accuracy(y, skm.labels_))
	print("Normalized Mutual Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, skm.labels_))
	print("Adjusted Rand-Index: %.3f" 
	   % metrics.adjusted_rand_score(y, skm.labels_))
	print("Adjusted Mututal Information: %.3f" 
	   % metrics.adjusted_mutual_info_score(y, skm.labels_))
	print("Normalized Mututal Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, skm.labels_))
	print("Silhouette Coefficient (euclidean): %0.3f" 
	   % metrics.silhouette_score(x, skm.labels_, metric='euclidean'))
	print("Silhouette Coefficient (cosine): %0.3f" 
	   % metrics.silhouette_score(x, skm.labels_, metric='cosine'))
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, skm.labels_))
	print("Completeness: %0.3f" % metrics.completeness_score(y, skm.labels_))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, skm.labels_))

	print()

	table.append([
		'spherical k-means',
		eval.accuracy(y, skm.labels_),
		metrics.normalized_mutual_info_score(y, skm.labels_),
		metrics.adjusted_rand_score(y, skm.labels_),
		metrics.adjusted_mutual_info_score(y, skm.labels_),
		metrics.homogeneity_score(y, skm.labels_),
		metrics.completeness_score(y, skm.labels_),
		metrics.v_measure_score(y, skm.labels_),
		metrics.silhouette_score(x, skm.labels_, metric='cosine'),
		metrics.silhouette_score(x, skm.labels_, metric='euclidean')])
	


	###############################################################################
	# Spherical K-Means plus clustering
	skmp = SphericalKmeansPlus(n_clusters=n_clusters, max_iter=300, init='k-means++', random_state=1000)
	
	print("Clustering with %s" % skmp)
	skmp.fit(csr_matrix(x))
	print()
	
	print("Accuracy: %.3f" % eval.accuracy(y, skmp.labels_))
	print("Normalized Mutual Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, skmp.labels_))
	print("Adjusted Rand-Index: %.3f" 
	   % metrics.adjusted_rand_score(y, skmp.labels_))
	print("Adjusted Mututal Information: %.3f" 
	   % metrics.adjusted_mutual_info_score(y, skmp.labels_))
	print("Normalized Mututal Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, skmp.labels_))
	print("Silhouette Coefficient (euclidean): %0.3f" 
	   % metrics.silhouette_score(x, skmp.labels_, metric='euclidean'))
	print("Silhouette Coefficient (cosine): %0.3f" 
	   % metrics.silhouette_score(x, skmp.labels_, metric='cosine'))
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, skmp.labels_))
	print("Completeness: %0.3f" % metrics.completeness_score(y, skmp.labels_))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, skmp.labels_))

	print()

	table.append([
		'spherical k-means++',
		eval.accuracy(y, skmp.labels_),
		metrics.normalized_mutual_info_score(y, skmp.labels_),
		metrics.adjusted_rand_score(y, skmp.labels_),
		metrics.adjusted_mutual_info_score(y, skmp.labels_),
		metrics.homogeneity_score(y, skmp.labels_),
		metrics.completeness_score(y, skmp.labels_),
		metrics.v_measure_score(y, skmp.labels_),
		metrics.silhouette_score(x, skmp.labels_, metric='cosine'),
		metrics.silhouette_score(x, skmp.labels_, metric='euclidean')])


	###############################################################################
	# Mixture of von Mises Fisher clustering (soft)
	vmf_soft = VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='soft', n_init=20, random_state=1000)
	
	print("Clustering with %s" % vmf_soft)
	vmf_soft.fit(x)
	print()
	print('weights: {}'.format(vmf_soft.weights_))
	print('concentrations: {}'.format(vmf_soft.concentrations_))

	print("Accuracy: %.3f" % eval.accuracy(y, vmf_soft.labels_))
	print("Normalized Mutual Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, vmf_soft.labels_))
	print("Adjusted Rand-Index: %.3f" 
	   % metrics.adjusted_rand_score(y, vmf_soft.labels_))
	print("Adjusted Mututal Information: %.3f" 
	   % metrics.adjusted_mutual_info_score(y, vmf_soft.labels_))
	print("Normalized Mututal Information: %.3f" 
	   % metrics.normalized_mutual_info_score(y, vmf_soft.labels_))
	print("Silhouette Coefficient (euclidean): %0.3f" 
	   % metrics.silhouette_score(x, vmf_soft.labels_, metric='euclidean'))
	print("Silhouette Coefficient (cosine): %0.3f" 
	   % metrics.silhouette_score(x, vmf_soft.labels_, metric='cosine'))
	print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, vmf_soft.labels_))
	print("Completeness: %0.3f" % metrics.completeness_score(y, vmf_soft.labels_))
	print("V-measure: %0.3f" % metrics.v_measure_score(y, vmf_soft.labels_))

	print()

	table.append([
		'movMF-soft',
		eval.accuracy(y, vmf_soft.labels_),
		metrics.normalized_mutual_info_score(y, vmf_soft.labels_),
		metrics.adjusted_rand_score(y, vmf_soft.labels_),
		metrics.adjusted_mutual_info_score(y, vmf_soft.labels_),
		metrics.homogeneity_score(y, vmf_soft.labels_),
		metrics.completeness_score(y, vmf_soft.labels_),
		metrics.v_measure_score(y, vmf_soft.labels_),
		metrics.silhouette_score(x, vmf_soft.labels_, metric='cosine'),
		metrics.silhouette_score(x, vmf_soft.labels_, metric='euclidean')])


	# ###############################################################################
	# # Mixture of von Mises Fisher clustering (hard)
	# vmf_hard = VonMisesFisherMixture(n_clusters=n_clusters, posterior_type='hard', random_state=2024)
	
	# print("Clustering with %s" % vmf_hard)
	# vmf_hard.fit(x)

	# print()
	# print('weights: {}'.format(vmf_hard.weights_))
	# print('concentrations: {}'.format(vmf_hard.concentrations_))

	# print("Accuracy: %.3f" % eval.accuracy(y, vmf_hard.labels_))
	# print("Normalized Mutual Information: %.3f" 
	#    % metrics.normalized_mutual_info_score(y, vmf_hard.labels_))
	# print("Adjusted Rand-Index: %.3f" 
	#    % metrics.adjusted_rand_score(y, vmf_hard.labels_))
	# print("Adjusted Mututal Information: %.3f" 
	#    % metrics.adjusted_mutual_info_score(y, vmf_hard.labels_))
	# print("Normalized Mututal Information: %.3f" 
	#    % metrics.normalized_mutual_info_score(y, vmf_hard.labels_))
	# print("Silhouette Coefficient (euclidean): %0.3f" 
	#    % metrics.silhouette_score(x, vmf_hard.labels_, metric='euclidean'))
	# print("Silhouette Coefficient (cosine): %0.3f" 
	#    % metrics.silhouette_score(x, vmf_hard.labels_, metric='cosine'))
	# print("Homogeneity: %0.3f" % metrics.homogeneity_score(y, vmf_hard.labels_))
	# print("Completeness: %0.3f" % metrics.completeness_score(y, vmf_hard.labels_))
	# print("V-measure: %0.3f" % metrics.v_measure_score(y, vmf_hard.labels_))

	# print()

	# table.append([
	# 	'movMF-hard',
	# 	eval.accuracy(y, vmf_hard.labels_),
	# 	metrics.normalized_mutual_info_score(y, vmf_hard.labels_),
	# 	metrics.adjusted_rand_score(y, vmf_hard.labels_),
	# 	metrics.adjusted_mutual_info_score(y, vmf_hard.labels_),
	# 	metrics.homogeneity_score(y, vmf_hard.labels_),
	# 	metrics.completeness_score(y, vmf_hard.labels_),
	# 	metrics.v_measure_score(y, vmf_hard.labels_),
	# 	metrics.silhouette_score(x, vmf_hard.labels_, metric='cosine'),
	# 	metrics.silhouette_score(x, vmf_hard.labels_, metric='euclidean')])


	###############################################################################
	# Print all results in table
	headers = [
		f'{args.word_emb} - {args.dim_red}',
		'Accuracy',
		'Norm MI',
		'Adj Rand',
		'Adj MI',
		'Homogeneity',
		'Completeness',
		'V-Measure',
		'Silhouette (cos)',
		'Silhouette (euc)']

	print(tabulate(table, headers, tablefmt="fancy_grid"))