from os import listdir,path,mkdir
from collections import Counter,defaultdict
import numpy as np

word_count_in_class = [] #word_count_in_class[k].get(word) 第k类中单词word出现的次数
total_word_in_class = [] #total_word_in_class[k] 第k类中所有单词出现的次数
doc_class_count = [] #doc_class_count[k] 第k类文档总数
doc_count = 0 #训练集所有文档总数
lexicon = set() #词典包括训练集出现的所有单词

#train
data = listdir('train')
for i in range(len(data)):
	tf = Counter() #用于计算词频
	word_count = 0
	cateDir = 'train/' + data[i]
	cateList = listdir(cateDir)
	doc_count += len(cateList)
	doc_class_count.append(len(cateList))
	for j in range(len(cateList)): 
		words = open('train/' + data[i] + '/' + cateList[j], encoding='ISO-8859-1').readlines()
		for word in words:
			word_count += 1
			lexicon.add(word)
			tf[word.split()[0]] += 1 #统计该类中每个词出现的次数
	total_word_in_class.append(word_count)
	d = dict(tf)
	word_count_in_class.append(d)

total_words_in_all_classes = len(lexicon)
print('训练集文档数量：',doc_count,' 词典大小：',total_words_in_all_classes)
print('正在分类...')
#print(total_words_in_all_classes)
#print(word_count_in_class[0].get('mathew'))
#print(total_word_in_class[0])

#predict
dir = listdir('test')
count = 0 #测试集文档总数
acc = 0 #分类正确的文档数
for i in range(len(dir)):
	cateDir = 'test/' + dir[i]
	cateList = listdir(cateDir)
	for j in range(len(cateList)):
		count += 1
		words = open('test/' + data[i] + '/' + cateList[j], encoding='ISO-8859-1').readlines()
		max = -1000000
		cate = -1
		for k in range(len(listdir('train'))):
			score = 0
			for word in words:
				if word_count_in_class[k].get(word.split()[0]) == None:
					tf = 0
				else:
					tf = word_count_in_class[k].get(word.split()[0])
				score += np.log((tf + 1) / (total_word_in_class[k] + total_words_in_all_classes))
			score += np.log(doc_class_count[k] / doc_count) #计算后验概率P(d|C)
			#print(score)
			if score > max:
				max = score
				cate = k
		if (cate == i):
			acc += 1
print('测试集文档数量：', count,' 分类正确数：', acc, ' 准确率为：', acc / count)
