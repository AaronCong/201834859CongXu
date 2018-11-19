from os import listdir,path,mkdir
import shutil,os
import nltk
from nltk.corpus import stopwords
import re

# def processFile(data_cate, data_name):
# 	srcFile = 'data_origin/' + data_cate + '/' + data_name
# 	targetFile = 'data/' + data_cate + '/' + data_name
# 	fw = open(targetFile, 'w', encoding='ISO-8859-1')	
# 	dataList = open(srcFile, encoding='ISO-8859-1').readlines() #使用utf-8编码会报错
# 	for line in dataList:
# 		resLine = lineProcess(line)
# 		for word in resLine:
# 			fw.write(word + '\n')
# 	fw.close()

# def lineProcess(line):
# 	line = re.sub(r'[^a-zA-Z]'," ",line)
# 	stopwords = nltk.corpus.stopwords.words('english') #去停用词
# 	porter = nltk.PorterStemmer() #提取词干
# 	splitter = re.compile('[^a-zA-z]') #过滤非字母字符
# 	words = [porter.stem(word.lower()) for word in line.split(" ") if len(word) > 0 and word.lower() not in stopwords]
# 	return words

# data_origin = listdir('data_origin') #原始数据目录
# for i in range(len(data_origin)):
# 	dataDir = 'data_origin/' + data_origin[i]
# 	dataList = listdir(dataDir)
# 	newDir = 'data/' + data_origin[i] #保存预处理后的数据目录
# 	if path.exists(newDir) == False:
# 		mkdir(newDir)
# 	else:
# 		print('%s exists', newDir)
# 	for j in range(len(dataList)):
# 		processFile(data_origin[i], dataList[j])

data = listdir('data')
for i in range(len(data)):
	dataDir = 'data/' + data[i]
	dataList = listdir(dataDir)
	newDir = 'train/' + data[i]
	if not path.exists(newDir):
		mkdir(newDir)
	for j in range(int(len(dataList)*0.8)):
		srcData = 'data/' + data[i]  + '/' + dataList[j]
		dstData = 'train/' + data[i] + '/' + dataList[j]
		shutil.copyfile(srcData, dstData)
	newDir = 'test/' + data[i]
	if not path.exists(newDir):
		mkdir(newDir)	
	for j in range(int(len(dataList)*0.8),len(dataList)):
		srcData = 'data/' + data[i] + '/' + dataList[j]
		dstData = 'test/' + data[i] + '/' + dataList[j]
		shutil.copyfile(srcData, dstData)