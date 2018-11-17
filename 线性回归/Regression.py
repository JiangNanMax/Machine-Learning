#!/usr/bin/python
# -*- coding utf-8 -*-
# Project: Regression
# Author: jiangnan 
# Mail: jiangnanmax@gmail.com
# Date: 2018/10/13

import numpy as np

def loadTrainData(filename):
    """
    函数说明：
        加载训练数据
    :param filename:
        文件名
    :return:
        xArray - x数据集，即为每个训练样本的特征参数
        yArray - y数据集，即为每个训练样本的年龄
    """
    featNum = len(open(filename).readline().split(',')) - 2 # 特征参数的个数，其中舍掉了第一个性别特征

    file = open(filename)
    xArray = []
    yArray = []
    for line in file.readlines():
        tempLine = line.strip().split(',')
        '''
        if tempLine[0] == 'M':
            tempLine[0] = '1'
        elif tempLine[0] == 'F':
            tempLine[0] = '-1'
        else:
            tempLine[0] = '0'
        '''
        del(tempLine[0])

        xArr = []
        for i in range(featNum):
            xArr.append(float(tempLine[i]))
        xArray.append(xArr)
        yArray.append(float(tempLine[-1]))

    return xArray, yArray

def loadTestData(filename):
    """
    函数说明：
        加载测试数据
    :param filename:
        文件名
    :return:
        xArray - x数据集，即为每个测试样本的特征参数
    """
    featNum = len(open(filename).readline().split(',')) - 1 # 特征参数的个数，其中舍掉了第一个性别特征

    file = open(filename)
    xArray = []
    for line in file.readlines():
        tempLine = line.strip().split(',')
        '''
        if tempLine[0] == 'M':
            tempLine[0] = '1'
        elif tempLine[0] == 'F':
            tempLine[0] = '-1'
        else:
            tempLine[0] = '0'
        '''
        del(tempLine[0])

        xArr = []
        for i in range(featNum):
            xArr.append(float(tempLine[i]))
        xArray.append(xArr)

    return xArray

def lwlRegression(testPoint, xArr, yArr, k=1.0):
    """
    函数说明：
        使用局部加权线性回归计算回归系数w
    :param testPoint:
        测试样本
    :param xArr:
        x训练数据集
    :param yArr:
        y训练数据集
    :param k:
        高斯核的k值，默认为1.0，可自定义
    :return:
        testPoint * ws - 计算得到的系数w对测试样本的预测值
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    for i in range(m):
        diffMat = testPoint - xMat[i, :]
        weights[i, i] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("不能求逆!")
        return

    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def RegressionTest(testArr, xArr, yArr, k=1.0):
    """
    函数说明：
        局部加权线性回归测试
    :param testArr:
        测试数据集
    :param xArr:
        x训练数据集
    :param yArr:
        y训练数据集
    :param k:
        高斯核的k值，默认为1.0，可自定义
    :return:
        yHat - 测试集合的所有预测值
    """
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlRegression(testArr[i], xArr, yArr, k)
    return yHat

def main():
    """
    函数说明：
        主函数，综合调用上述功能函数完成工作
    """
    trainX, trainY = loadTrainData('train.txt');
    testX = loadTestData('test.txt')

    yHat = RegressionTest(testX, trainX, trainY, 0.1)

    # 输出预测结果
    for i in yHat:
        print(i)

if __name__ == '__main__':
    main()


