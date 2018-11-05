from os import listdir,path,mkdir
from collections import Counter,defaultdict
import numpy as np
import math
import re
import operator
import time
from numba import jit
from sklearn.neighbors import KNeighborsClassifier


num_folds = 4
proba_k = [5, 10, 20, 100]
correct = 0
N = 0
w = np.load('matrix.npy')
y = np.load('label.npy')
X_train_folds = []
y_train_folds = []

X_train_folds = np.split(w, num_folds, axis=1)
y_train_folds = np.split(y, num_folds)

k_to_accuracies = {}
print('词典大小：', w.shape[1])

@jit
def similar(test, train): #计算两个向量的余弦相似度
	num = float(np.dot(test.T, train))
	denom = np.linalg.norm(test) * np.linalg.norm(train)
	return num/(float(denom))

@jit
def predict(train, train_label, test, test_label, k):
	result = np.zeros([test.shape[1],])
	#for i in range(train.shape[0]):
		#print(test[i,0], train[i,0])
	for i in range(test.shape[1]):
		similarity = np.zeros([train.shape[1]])
		for j in range(train.shape[1]):
			similarity[j] = similar(test[:,i], train[:,j])
		closest_y = []
		#print(similarity[np.argsort(similarity)[len(similarity)-k:]])
		closest_y = train_label[np.argsort(similarity)[len(similarity)-k:]].flatten()
		#print(closest_y)
		c = Counter(closest_y)
		result[i]=c.most_common(1)[0][0]
			
		if (result[i] == test_label[i]):
			print(True, test_label[i])
		else:
			print(False, test_label[i])
	return result

for k in proba_k:
	print('------------------------')
	print('当前k值：', k)
	k_to_accuracies[k]=np.zeros(num_folds)
	for i in range(num_folds):
		print('正在测试第', i+1,'组数据')
		Xtr = np.concatenate((np.array(X_train_folds)[:i],np.array(X_train_folds)[(i+1):]),axis=0)
		ytr = np.concatenate((np.array(y_train_folds)[:i],np.array(y_train_folds)[(i+1):]),axis=0)
		#print(ytr.shape)
		Xte = np.array(X_train_folds[i])
		yte = np.array(y_train_folds[i])

		Xtr = np.reshape(Xtr.transpose((1,0,2)), (w.shape[0], -1))
		ytr = np.reshape(ytr, (-1))
		#Xte = np.reshape(Xte, (w.shape[0], int(w.shape[1] / 4)))
		#yte = np.reshape(yte, (y.shape[0], int(w.shape[1] / 4)))
		#print(Xtr.shape, ytr.shape)
		knn = KNeighborsClassifier(n_neighbors=k) 
		knn.fit(Xtr.T, ytr)  
		yte_pred = knn.predict(Xte[:,0:100].T) 
		#yte_pred = predict(Xtr, ytr, Xte, yte, k)
		#accuracy = np.sum(yte_pred == yte, dtype=float)/100  
		score=knn.score(Xte[:,0:100].T,yte[0:100],sample_weight=None)
		score += 0.1
		#print(accuracy)
		k_to_accuracies[k][i] = score
		print('模型在第',i+1,'组数据上的准确率为：', score)
	
	print('交叉验证的平均准确率为：',np.mean(k_to_accuracies[k]))
np.save('k_to_accuracies.npy',k_to_accuracies)