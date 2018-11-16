#!/usr/bin/python
# -*- coding utf-8 -*-
# Project: NB
# Author: jiangnan 
# Mail: jiangnanmax@gmail.com
# Date: 2018/9/26

import numpy as np

def loadFile(filename):
    """
    函数说明：
        加载数据文件
    :param filename:
        文件名
    :return:
        contentList - 切分邮件内容得到的词条
        classVec - 类别标签向量
    """
    file = open(filename)

    contentList = []
    classVec = []

    contents = file.readlines()
    for line in contents:
        content = line.strip('\n').split(' ')   #以空格为分割符，切分邮件的内容，得到该邮件对应的词条
        classVec.append(int(content[0]))    #取出邮件的类别标签
        del(content[0])     #删掉词条中的类别标签
        contentList.append(content)

    return contentList, classVecx


def createVocabList(dataSet):
    """
    函数说明：
        根据训练数据，生成一个词汇表
    :param dataSet:
        切分所有邮件得到的词条
    :return:
        list(vocabSet) - 使用训练数据生成的不重复的词汇表
    """
    vocabList = set([])  #创建一个空集合
    for content in dataSet:
        vocabList = vocabList | set(content)   #通过取并集的方式去重，扩充词汇表

    return list(vocabList)   #以list的形式返回词汇表


def Words_to_Vec(vocabList, wordsSet):
    """
    函数说明：
        根据vocabList词汇表，将每个wordsSet词条向量化，向量的每个值为1或0，分别表示该词有或者没有在词汇表中出现
    :param vocabList:
        词汇表
    :param inputSet:
        切分每封邮件得到的词条
    :return:
        词条向量
    """
    returnVec = [0] * len(vocabList)

    for word in wordsSet:   #判断每个词是否在词汇表中出现
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1    #在词汇表中出现的话则该词对应的位置标记为1

        else:
            print("The word %s is not in the VocabList!" % word)

    return returnVec


def trainNB(trainMat, trainLabel):
    """
    函数说明：
        朴素贝叶斯分类训练函数
    :param trainMat:
        训练文档，即Words_to_Vec函数返回的词向量构成的矩阵
    :param trainLabel:
        训练数据的类别标签，即loadFile函数返回的classVec
    :return:
        p0Vec - 侮辱类的条件概率数组
        p1Vec - 非侮辱类的条件概率数组
        pNotAbusive - 文档属于侮辱类的概率
    """
    numTraindocs = len(trainMat)    #训练集的数量
    numWords = len(trainMat[0])     #每个词条向量的长度

    pNotAbusive = sum(trainLabel) / float(numTraindocs)    #文档属于非侮辱类的概率

    p0Num = np.ones(numWords)   #创建numpy.ones数组,词条出现数初始化为1，拉普拉斯平滑方法
    p1Num = np.ones(numWords)
    p0Denom = 2.0               ##分母初始化为2,拉普拉斯平滑方法
    p1Denom = 2.0

    for i in range(numTraindocs):
        if trainLabel[i] == 1:
            p1Num += trainMat[i]    #统计属于非侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)···
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]    #统计属于侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
            p0Denom += sum(trainMat[i])

    p1Vec = np.log(p1Num / p1Denom) #取对数
    p0Vec = np.log(p0Num / p0Denom)

    return p0Vec, p1Vec, pNotAbusive


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass0):
    """
    函数说明：
        朴素贝叶斯分类函数
    :param vec2Classify:
        待分类的词条向量
    :param p0Vec:
        侮辱类的条件概率数组
    :param p1Vec:
        非侮辱类的条件概率数组
    :param pClass0:
        文档属于侮辱类的概率
    :return:
        0 - 文档属于侮辱类
        1 - 文档属于分侮辱类
    """
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass0)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass0)
    if p1 > p0:
        return 1
    else:
        return 0

def main():
    trainList, trainLabel = loadFile('spam_train.txt')  #处理训练数据
    VocabList = createVocabList(trainList)  #生成词汇表

    '''
    #由于数据集较大，生成的词汇表内容较多，因此我把生成的词汇表写出到txt文件，后续可以直接调用
    file = open('VocabList.txt', 'w')
    file.write(str(VocabList))
    file.close()
    '''

    trainMat = []
    cnt = 0     #用来标记处理到第几组数据，在处理完每组数据后就会加1输出

    for train in trainList:     #生成训练集矩阵
        trainMat.append(Words_to_Vec(VocabList, train))
        cnt += 1
        print("当前处理到第%s组训练数据." % cnt)

    '''
    #由于数据集较大，生成训练集矩阵较慢，内容也较多，因此我把生成的矩阵写出到txt文件，后续可以直接调用
    file = open('trainMat.txt', 'w')
    file.write(str(trainMat))
    file.close()
    '''

    p0V, p1V, pAb = trainNB(np.array(trainMat), np.array(trainLabel))   #使用训练集矩阵训练分类器

    testList, testLabel = loadFile('spam_test.txt')     #处理测试数据
    resultLabel = []

    cnt = 0

    for test in testList:   #分类测试数据，将分类的标签放入resultLabel
        doc = np.array(Words_to_Vec(VocabList, test))

        if classifyNB(doc, p0V, p1V, pAb):
            resultLabel.append(1)
        else:
            resultLabel.append(0)

        cnt += 1
        print("当前处理到第%s组测试数据." % cnt)

    cc = 0  #分类正确的个数
    for i in range(len(testLabel)):     #对比分类标签和真实标签
        if testLabel[i] == resultLabel[i]:
            cc += 1

    print('预测准确率：' + str(100 * cc / float(len(testLabel))) + '%')   #计算准确率


if __name__ == '__main__':
    main()