# -*- coding: utf-8 -*-
"""
# @Time    : 2019/11/21
# @Author  : zhanzecheng/片刻/DrDavidS
# @File    : demo.py.py
# @Software: PyCharm
"""
import os

import jieba

from model import TrieNode
from utils import get_stopwords, load_dictionary, generate_ngram, save_model, load_model
from config import basedir


def load_data(filename, stopwords):
    """
    按行读取信息

    :param filename:
    :param stopwords:
    :return: 二维数组,[[句子1分词list], [句子2分词list],...,[句子n分词list]]
    """
    data = []
    with open(filename, 'r',  encoding='utf-8') as f:
        for line in f:
            word_list = [x for x in jieba.cut(
                line.strip(), cut_all=False) if x not in stopwords]
            data.append(word_list)
    return data


def load_data_2_root(data):
    print('------> 插入节点')
    for word_list in data:
        # tmp 表示每一行自由组合后的结果（n gram）
        # tmp: [['它'], ['是'], ['小'], ['狗'], ['它', '是'], ['是', '小'], ['小', '狗'], ['它', '是', '小'], ['是', '小', '狗']]
        ngrams = generate_ngram(word_list, 3)
        for d in ngrams:
            root.add(d)
    print('------> 插入成功')


if __name__ == "__main__":
    root_name = basedir + "/data/root.pkl"
    stopwords = get_stopwords()
    if os.path.exists(root_name):
        root = load_model(root_name)
    else:
        dict_name = basedir + '/data/dict.txt'
        word_freq = load_dictionary(dict_name)
        root = TrieNode('*', word_freq)
        save_model(root, root_name)

    # 加载新的文章
    filename = 'data/demo.txt'
    data = load_data(filename, stopwords)
    # 将新的文章插入到Root中
    load_data_2_root(data)

    # 新增词语
    new_words = []
    # 定义取topN个
    topN = 100
    result, add_word = root.find_word(topN)
    # 如果想要调试和选择其他的阈值，可以print result来调整
    # print("\n----\n", result)
    print("\n----\n", '增加了 %d 个新词, 词语和得分分别为: \n' % len(add_word))
    print('#############################')
    for word, score in add_word.items():
        print(word + ' ----> ', score)
        new_words.append(word)
    print('#############################')

    # 保存新增词语
    if os.path.exists("./result_new_words/new_words.txt"):
        print("已经存在文件，删除")
        os.remove("./result_new_words/new_words.txt")
    mylist = new_words

    with open("./result_new_words/new_words.txt", 'w', encoding='utf-8') as f:
        for var in mylist:
            f.writelines(var)
            f.write('\n')
    f.close()

    # 前后效果对比
    test_sentence = '小微企业没有达到要交税的标准，免征的增值税是记入营业外收入交企税吗'
    print('添加前：')
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence,
                                                 cut_all=False) if x not in stopwords]))

    for word in add_word.keys():
        jieba.add_word(word)
    print("添加后：")
    print("".join([(x + '/ ') for x in jieba.cut(test_sentence,
                                                 cut_all=False) if x not in stopwords]))
