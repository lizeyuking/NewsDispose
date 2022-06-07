# coding=utf-8
import re
import time
import pandas as pd
import numpy as np
from collections import Counter
import jieba.analyse


def Chinese(text):      #处理函数
    cleaned = re.findall(r'[\u4e00-\u9fa5]+', text)  #返回列表,用正则表达式筛选出中文
    cleaned = ''.join(cleaned)                      #拼接成字符串
    return cleaned
#------------------------------------中文分词------------------------------------
cut_words = ""
f = open('数据集\\guanchazhe.csv','r',encoding='utf-8') #导入要处理的原始数据文件
data = pd.read_csv(f).astype(str)
record_num = int(data.describe().iloc[0,0])
f.close()
jieba.analyse.set_stop_words('stop_words\\total_stop_words.txt')   # 停用词词典
data['Tfidf关键词'] = 0
#遍历每行review进行分词
for i in range(record_num):
    record = data.iloc[i,:]     #第i行的所有数据
    comment = record['review']  #标头为revie的数据
    #comment = Chinese(comment)  # 清洗一下句子，只保留中文字符
    #seg_list = jieba.cut(comment, cut_all=False)    # 精确模式分词
    # print(" ".join(seg_list))
    #cut_words += ("".join(seg_list))
    # 提取主题词 返回的词频其实就是TF-IDF
    jieba_list = jieba.analyse.extract_tags(comment,
                                          topK=3,
                                          allowPOS=('a', 'e', 'n', 'nr', 'ns', 'v'))  # 词性 形容词 叹词 名
    keywords = ''
    for key in jieba_list:
        if keywords != '':
            keywords += "、"
        keywords += key
    keywords = ''.join(keywords)

    # keywords = jieba.analyse.extract_tags(cut_words,
    #                                       topK=3,
    #                                       withWeight=True,
    #                                       allowPOS=('a', 'e', 'n', 'nr', 'ns', 'v'))  # 词性 形容词 叹词 名词 动词
    data.iloc[i,-1] = keywords
    #record.to_csv("数据集\\微博.csv", index=False, sep=',') # 将新增的列数据，增加到原始数据中
    #pd.DataFrame(keywords, columns=['Tfidf关键词','重要性']).to_csv('数据集\\微博.csv')  # csv格式输出

    cut_words = ' '
#print("cut_words:",cut_words)
# jieba.load_userdict("userdict.txt")              # 自定义词典
data.to_csv("数据集\\观察者网新闻.csv") # 将新增的列数据，增加到原始数据中
