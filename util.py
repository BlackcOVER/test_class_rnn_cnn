# author: lifan
# create date: 2018-12-07
# update date: 2018-12-11
# description: 工具函数库
# help: 对应目录见UtilOutlook

import numpy as np
import tensorflow.contrib.keras as keras
from collections import Counter


# 数据batch生成器
def batch_iter(x, y, batch_size=64):
    '''
    :param x: 样本数据，array[None, weight, height, None]
    :param y: 标签数据，array[None, class_num]
    :param batch_size: batch大小， int
    :return: x_batch, y_batch
    '''
    data_len = len(x)
    # batch数量
    num_batch = int((data_len - 1) / batch_size) + 1
    # 数据混排
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


# 映射 字/词 到 id
def process_text(contents, labels, dict_vocab, dict_label, max_length=600, kind='word'):
    '''
    :param contents:  文本 str
    :param labels:  标签 str
    :param dict_vocab: 词/字 字典  key：word|vocab, value: id
    :param dict_label: label 字典  key: label, valur: id
    :param max_length: 保留词/字的最大数量
    :param kind: 文本元素模式 word为字 vocab为词
    :return: 文本id序列list， label one_hot编码
    '''
    # 字/词方式切割文本
    contents = [list(content) if kind == 'word' else content.split(' ') for content in contents]
    # 将文本元素映射为id
    contents_id = list(map(lambda content: [dict_vocab[x] for x in content], contents))
    # 将label映射为id
    label_id = [dict_label[label] for label in labels]
    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = keras.preprocessing.sequence.pad_sequences(contents_id, max_length)
    # 将标签转换为one-hot表示
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(dict_label))

    return x_pad, y_pad


# 建立词汇表
def build_vocab(data_train, vocab_dir, vocab_size=5000):
    '''
    :param data_train:  训练样本数据 每条数据是多个字组成的列表
    :param vocab_dir:  保存词的路径
    :param vocab_size:  保存词的大小
    :return:  出现次数最多的词vocab
    '''
    all_data = []
    for content in data_train:
        all_data.extend(content)
    # 计数器
    counter = Counter(all_data)
    # 返回出现次数最多的词
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    # 将此写入文件
    with open(vocab_dir, 'w', encoding='utf8') as vocab_file:
        print('\n'.join(words) + '\n', file=vocab_file)


# 生成词汇：id映射字典
def read_vocab(vocab_dir):
    '''
    :param vocab_dir: 词汇文件
    :return:  词汇列表 list ； 词汇id字典{word: id}
    '''
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir, 'r') as vocab_file:
        # 如果是py2 则每个值都转化为unicode
        words = [x.strip() for x in vocab_file]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# 生成分类：id映射字典
def read_category(labels):
    '''
    :param labels: labels列表
    :return: 分类列表 list ； 分类id字典{category: id}
    '''
    categories = list(set(labels)).sort()
    print('kinds of categories: ', categories)
    cat_to_id = dict(zip(categories, range(len(categories))))
    return categories, cat_to_id