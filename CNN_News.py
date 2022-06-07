import tqdm
import jieba

import kashgari
from kashgari.embeddings import BERTEmbedding         #Kashgari 内置了bert预训练语言模型处理模块
# 使用 embedding 初始化模型
from kashgari.tasks.classification import BiLSTM_Model
from tensorflow.python import keras
from kashgari.callbacks import EvalCallBack
import  tensorflow as tf
import os
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def read_data_file(path):
    lines = open(path, 'r', encoding='utf-8').read().splitlines()
    x_list = []
    y_list = []
    for line in tqdm.tqdm(lines):
        rows = line.split('\t')
        if len(rows) >= 2:
            y_list.append(rows[0])
            x_list.append(list(jieba.cut('\t'.join(rows[1:]))))
        else:
            print(rows)
    return x_list, y_list

test_x, test_y = read_data_file('cnews/cnews.test.txt')
train_x, train_y = read_data_file('cnews/cnews.train.txt')
val_x, val_y = read_data_file('cnews/cnews.val.txt')
from tensorflow.keras.callbacks import TensorBoard
from kashgari.tasks.classification import CNN_Model
model = CNN_Model()
# Using TensorBoard record training process
tf_board = TensorBoard(log_dir='logs',
                       histogram_freq=5,
                       update_freq='batch')
model.fit(train_x, train_y, val_x, val_y,
          batch_size=128,
          epochs=1,
          callbacks=[tf_board])
model.save('cnn_classification_model')