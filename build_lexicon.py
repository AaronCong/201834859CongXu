from os import listdir,path,mkdir
from collections import Counter,defaultdict
import numpy as np
import math
import time
data = listdir('data')
tf = Counter() #用于计算词频
for i in range(len(data)):
	cateDir = 'data/' + data[i]
	cateList = listdir(cateDir)
	for j in range(int(len(cateList)*0.8)): #取前80%的文档构建词典
		words = open('data/' + data[i] + '/' + cateList[j], encoding='ISO-8859-1').readlines()
		for word in words:
			tf[word.split()[0]] += 1 #统计每个词出现的次数

lexicon = dict(tf)
for key, value in lexicon.items():
	if value < 10:
		del(tf[key]) #过滤掉出现10次以内的词
lexicon = set(tf)

N = 0 #文档数量
df = {} #文档频率
in_dict = [] #记录文档中出现的词典词
for i in range(len(data)):
	cateDir = 'data/' + data[i]
	cateList = listdir(cateDir)
	N += len(cateList)
	for j in range(int(len(cateList)*0.8)):
		article = open('data/' + data[i] + '/' + cateList[j], encoding='ISO-8859-1').readlines()
		article = [word.split()[0] for word in article]
		for word in set(article):#计算df
			if word in lexicon:
				if word in df.keys(): 
					df[word] += 1
				else:
					df[word] = 1


lexicon = list(lexicon)
lexicon.sort()
fw = open('aa.txt','w')
for line in lexicon:
	fw.write(str(line) + '\n')
fw.close()


w = np.zeros([len(lexicon), N]) #VSM矩阵
label = np.zeros([N,])
num = 0 #文档索引
tf = 0 #文档频率
for i in range(len(data)):
	print("构建完成{0}%".format((i + 1) * 5))
	cateDir = 'data/' + data[i]
	cateList = listdir(cateDir)
	for j in range(len(cateList)):
		article = open('data/' + data[i] + '/' + cateList[j], encoding='ISO-8859-1').readlines()
		article = [word.split()[0] for word in article]
		for word in article: 
			if word in lexicon: 
				tf = article.count(word) / len(article)
				w[lexicon.index(word), num] = tf * np.log(N / df[word])
		label[num] = i
		# for k in range(len(in_dict[num])):#为每个词典中的词计算tf×idf
		# 	word = in_dict[num][k]
		# 	tf = article.count(word)/len(article)
		# 	w[lexicon.index(word), num] = tf * np.log(N/(df[word]))
		num += 1

permutation = np.random.permutation(label.shape[0])
shuffled_dataset = w[:,permutation]
shuffled_label = label[permutation]
# print(N)
# print(df)
np.save('matrix.npy', shuffled_dataset)
np.save('label.npy', shuffled_label)