# -*- coding: utf-8 -*-
import jieba
import numpy as np
import re
import os

def get_word_vector(s1, s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """


    # 将标点符号替换成空白，即删除掉标点符号
    s1 = re.sub(r"[0-9\s+\.\,]|[。，]","", s1)
    s2 = re.sub(r"[0-9\s+\.\,]|[。，]","", s2)
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)

    list_word1 =(','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')
    return list_word1,list_word2

def get_vector(list_word1,list_word2):
    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    print(key_word)
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))

    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    print(word_vector1)
    print(word_vector2)
    return word_vector1, word_vector2


def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist1



if __name__ == '__main__':
    with open('orig.txt', encoding='UTF-8') as f1:
        s1 = f1.read()
        with open('orig_add.txt', encoding='UTF-8') as f2:
            s2 = f2.read()
        with open('orig_add1.txt', encoding='UTF-8') as f3:
            s3 = f3.read()
        with open('orig_add2.txt', encoding='UTF-8') as f4:
            s4 = f4.read()
            with open('orig_add3.txt', encoding='UTF-8') as f5:
                s5 = f5.read()
                with open('orig_add4.txt', encoding='UTF-8') as f6:
                    s6 = f6.read()
word1, word2 = get_word_vector(s1, s2)
vec1, vec2 = get_vector(word1, word2)
dist1 = cos_dist(vec1, vec2)
word1, word2 = get_word_vector(s1, s3)
vec1, vec2 = get_vector(word1, word2)
dist2=cos_dist(vec1, vec2)
word1, word2 = get_word_vector(s1, s4)
vec1, vec2 = get_vector(word1, word2)
dist3 = cos_dist(vec1, vec2)
word1, word2 = get_word_vector(s1, s5)
vec1, vec2 = get_vector(word1, word2)
dist4 = cos_dist(vec1, vec2)
word1, word2 = get_word_vector(s1, s6)
vec1, vec2 = get_vector(word1, word2)
dist5 = cos_dist(vec1, vec2)


with open('orig_similar.txt', 'w') as file:
    file.write(str(dist1))
    file.close()
    with open('orig_similar1.txt', 'w') as file1:
        file1.write(str(dist2))
        file1.close()
        with open('orig_similar2.txt', 'w') as file2:
            file2.write(str(dist3))
            file2.close()
            with open('orig_similar3.txt', 'w') as file3:
                file3.write(str(dist4))
                file3.close()
                with open('orig_similar4.txt', 'w') as file4:
                    file4.write(str(dist5))
                    file4.close()

                    print(dist5)