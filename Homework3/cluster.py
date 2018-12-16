
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import *
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import numpy as np
from time import time
import warnings
warnings.filterwarnings("ignore")

corpus = []
labels = []
with open('Tweets.txt', 'r') as f:
	lines = f.readlines()
	for line in lines:
		js = json.loads(line)
		corpus.append(js.get('text'))
		labels.append(js.get('cluster'))

vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')
X = vectorizer.fit_transform(corpus)
clusters = np.unique(labels).shape[0]
print('数据加载完成，共',X.shape[0],'条tweets')

print('正在使用K-means算法聚类...')
t = time()
km = KMeans(n_clusters=clusters, init='k-means++', max_iter = 100, n_init = 1).fit(X)
km.fit(X)
print('NMI值为',metrics.normalized_mutual_info_score(labels, km.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用AffinityPropagation算法聚类...')
t = time()
ap = AffinityPropagation().fit(X)
print('NMI值为',metrics.normalized_mutual_info_score(labels, ap.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用Mean-Shift算法聚类...')
t = time()
X_array = X.toarray()
ms = MeanShift(bandwidth=0.9, bin_seeding=True).fit(X_array)
print('NMI值为',metrics.normalized_mutual_info_score(labels, ms.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用Spectral Clustering算法聚类...')
t = time()
sc = SpectralClustering(n_clusters=clusters).fit(X)
print('NMI值为',metrics.normalized_mutual_info_score(labels, sc.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用DBSCAN算法聚类...')
t = time()
dbscan = DBSCAN(min_samples=1,metric='cosine').fit(X)
print('NMI值为',metrics.normalized_mutual_info_score(labels, dbscan.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用Ward hierachical clustering算法聚类...')
t = time()
ward = AgglomerativeClustering(n_clusters=clusters, linkage='ward').fit(X_array)
print('NMI值为',metrics.normalized_mutual_info_score(labels, ward.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用Agglomerative Clustering算法聚类...')
t = time()
ac = AgglomerativeClustering(n_clusters=clusters, linkage='complete').fit(X_array)
print('NMI值为',metrics.normalized_mutual_info_score(labels, ac.labels_),'所用时间：','%.2fs' % (time() - t))

print('正在使用Gaussian Mixture算法聚类...')
t = time()
gm = GaussianMixture(n_components=50).fit(X_array)
print('NMI值为',metrics.normalized_mutual_info_score(labels, gm.predict(X_array)),'所用时间：','%.2fs' % (time() - t))
