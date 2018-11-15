import numpy as np
import operator

def trainingFile2Matrix(filename):
    """
    函数说明：
        处理训练数据集
    :param filename:
        训练数据文件
    :return:
        returnMat - 处理得到的每一个训练样本的数据集合
        returnLabel - 每一个训练样本所属的类别标签集合
    """
    file = open(filename)
    content = file.readlines()

    lineCount = len(content)

    returnMat = np.zeros((lineCount, 4))
    returnLabel = []

    index = 0

    for line in content:
        line = line.strip()
        example = line.split(',')

        returnMat[index, : ] = example[0 : 4]
        index += 1
        returnLabel.append(example[4])

    return returnMat, returnLabel


def testFile2Matrix(filename):
    """
    函数说明：
        处理测试数据集
    :param filename:
        测试数据文件
    :return:
        returnMat - 处理得到的每一个测试样本的数据集合
    """
    file = open(filename)

    content = file.readlines()

    lineCount = len(content)

    returnMat = np.zeros((lineCount, 4))

    index = 0

    for line in content:
        line = line.strip()
        example = line.split(',')
        returnMat[index, : ] = example[0 : 4]
        index += 1

    return returnMat



def calculateDistance(train_example, test_example, example_length):
    """
    函数说明：
        计算训练样本和测试样本之间的欧几里德距离
    :param train_example:
        训练样本的数据
    :param test_example:
        测试样本的数据
    :param example_length:
        样本的属性长度
    :return:
        distance - 训练样本和测试样本之间的欧几里德距离
    """
    distance = 0.0
    for i in range(example_length):
        distance += pow(train_example[i] - test_example[i], 2)

    return distance

def get_K_Neighbors(trainingSet, trainingLabel, test_example, k):
    """
    函数说明：
        取得与测试样本距离最近的k个训练样本
    :param trainingSet:
        训练样本数据集
    :param trainingLabel:
        训练样本标签集
    :param test_example:
        测试样本
    :param k:
        即参数k
    :return:
        kNeighbors - 与测试样本最近的k个训练样本的集合
    """
    length = len(test_example)
    distances = []

    for i in range(len(trainingSet)):
        dis = calculateDistance(trainingSet[i], test_example, length)
        distances.append((trainingLabel[i], dis))

    distances.sort(key=operator.itemgetter(1))

    kNeighbors = []
    for i in range(k):
        kNeighbors.append(distances[i][0])

    return kNeighbors


def getReasult(kNeighbors):
    """
    函数说明：
        取得与测试样本距离最近的k个训练样本中的最公共类别
    :param kNeighbors:
        与测试样本最近的k个训练样本的集合
    :return:
        sortedLabel[0][0] - 预测该测试样本所属的类别
    """
    classLabel = {}
    for i in range(len(kNeighbors)):
        temp = kNeighbors[i]
        if temp in classLabel:
            classLabel[temp] += 1
        else:
            classLabel[temp] = 1

    sortedLabel = sorted(classLabel.items(), key=operator.itemgetter(1), reverse=True)
    return sortedLabel[0][0]

def getAccuracy(testLabel, predictions):
    """
    函数说明：
        计算预测的准确率
    :param testLabel:
        测试数据所属的真实类别
    :param predictions:
        预测测试数据所属的类别
    :return:
        (cnt / float(len(testLabel))) * 100.0 - 准确率
    """
    cnt = 0

    for i in range(len(testLabel)):
        if(testLabel[i] == predictions[i]):
            cnt += 1

    return (cnt / float(len(testLabel))) * 100.0


def getNormolization(dataSet):
    """
    函数说明：
        对数据进行归一化
    :param dataSet:
        数据集合
    :return:
        normDataSet - 归一化后的数据集合
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)

    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))

    m = dataSet.shape[0]

    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet / np.tile(ranges, (m,1))

    return normDataSet

def write2File(filename, resultSet):
    """
    函数说明：
        将测试结果写入文件
    :param filename:
        要写入的文件
    :param resultSet:
        测试结果集合
    """
    with open(filename, "r", encoding="utf-8") as f_read:
        content = f_read.readlines()

    #print(content)

    index = 0
    length = len(resultSet)


    with open(filename, "w", encoding="utf-8") as f_write:
        for i in range(length):
            str = ''
            temp = content[i].strip('\n')

            str = temp + ',' + resultSet[i] + '\n'
            index += 1

            f_write.write(str)


def classify():
    """
    函数说明：
        综合调用前面的功能函数，实现KNN算法的所有步骤
    """

    #自定义测试
    trainingMat, trainingLabel = trainingFile2Matrix("train.txt")

    testMat = testFile2Matrix("test.txt")

    norm_trainingMat = getNormolization(trainingMat)
    norm_testMat = getNormolization(testMat)

    #print(norm_trainingMat)
    #print()
    #print(norm_testMat)


    result = []

    for i in range(len(testMat)):
        kNeighbors = get_K_Neighbors(norm_trainingMat, trainingLabel, norm_testMat[i], 3)
        #print(kNeighbors)
        #print()jignn
        result.append(getReasult(kNeighbors))

    #print(result)
    #print(len(result))

    #print("预测准确率：" + str(getAccuracy(testLabel, result)))

    #print()
    write2File("test.txt", result)



if __name__ == "__main__":
    classify()