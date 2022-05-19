#-*- coding: utf-8 -*-

# 识别手写的数字

import numpy as np
import operator
import time
import os
from PIL import Image
import matplotlib.pyplot as plt


# KNN算法
def classify(inputPoint, dataSet, labels, k):
    """
    params:
        inputPoint: 待测试集, 1行n列的list（n为特征个数）
        dataSet：训练集, m×n的list（m为训练数据的数量）
        labels: 训练集结果, 一个m行1列的list，它对应着dataSet中每一行的标签，即预测的结果
        k: 表示knn的中k的值
    """
    # print("inputPoint:%s" % np.array(inputPoint).shape) # 1024
    # print("dataSet:", np.array(dataSet).shape)    # (1934, 1024)
    # print("labels:%s" % np.array(labels).shape)   # 1934

    dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数
    #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = np.tile(inputPoint, (dataSetSize, 1)) - dataSet  #样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2                    #差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
    distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表
    #选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[ sortedDistIndicies[i] ]
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    #按classCount字典的第2个元素（即类别出现的次数）从大到小排序
    sortedClassCount = sorted(classCount.items(), key=lambda i:i[1], reverse=True) # sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]


# 文本向量化 32x32 -> 1x1024
def img2vector(filename):
    returnVect = []
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect


# 从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


# 从文件名中解析分类数字
def classnumCut(fileName):
    fileStr = fileName.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    return classNumStr


# 构建训练集数据向量，及对应分类标签向量
def trainingDataSet(train_path):
    hwLabels = []
    trainingFileList = os.listdir(train_path)           # 获取目录内容
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))                         # m维向量的训练集
    for i in range(m):
        fileNameStr = trainingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i, :] = img2vector('%s/%s' % (train_path, fileNameStr))
    return hwLabels, trainingMat


# 测试函数
def handwritingTest(test_path, train_path, k):
    hwLabels, trainingMat = trainingDataSet(train_path)   # 构建训练集
    testFileList = os.listdir(test_path)        # 获取测试集
    errorCount = 0.0                            # 错误数
    mTest = len(testFileList)                   # 测试集总样本数

    t1 = time.time()
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumStr = classnumCut(fileNameStr)  # 真实数字值
        vectorUnderTest = img2vector('%s/%s' % (test_path, fileNameStr))  # 预处理后的测试集

        # 调用knn算法进行测试
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

        if classifierResult != classNumStr:
            errorCount += 1.0

    print("\nthe total number of tests is: %d" % mTest)               # 输出测试总样本数
    print("the total number of errors is: %d" % errorCount)           # 输出测试错误样本数
    print("the total error rate is: %f" % (errorCount/float(mTest)))  # 输出错误率
    t2 = time.time()
    print("Cost time: %.2fmin, %.4fs."%((t2-t1)//60, (t2-t1)%60))     # 测试耗时

    return errorCount/float(mTest)   # 错误率



def transferPic2grey(originPath):
    """
    func:将彩色图片转成32×32的灰度图片
    :param originPath: 彩色图片路径
    :return:
    """
    I = Image.open(originPath)
    I = I.resize((32, 32))  # 将图片大小压缩至32×32
    L = I.convert('L')      # 将图片转换为灰度图片
    save_path = originPath.split('.')[-2] + '_grey.jpg'
    L.save(save_path)
    return save_path


def transferGrey2binary(originPath):
    """
    func:将灰度图片转成二进制文件，背景色（白）的灰度图片的二进制大约为170左右，而笔迹（黑）大约为50以内，
         因为训练集中数字用1表示，背景用0表示，因此将二进制大于100的都置为0，小于100的都置为1
    :param originPath:
    :return:
    """
    im = Image.open(originPath)
    # print(im.size)   # (32, 32)
    save_path = originPath.split('.')[-2] + "_.txt"
    with open(save_path, 'w') as f:
        for i in range(im.size[1]):
            for j in range(im.size[0]):
                color = im.getpixel((j, i))
                color = 0 if color > 100 else 1
                f.write(str(color))
            f.write("\n")
    return save_path



def testMyHandWriteImgRecoginze(test_pic_path, k):
    grey_origin_path = transferPic2grey(test_pic_path)
    grey_binary_path = transferGrey2binary(grey_origin_path)


    hwLabels, trainingMat = trainingDataSet(train_path='trainingDigits')   #构建训练集
    vectorUnderTest = img2vector(grey_binary_path)
    classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, k)
    print("result is: %d" % int(classifierResult))



if __name__ == "__main__":
    # k_error = []
    # k_range = range(1, 10)
    # for k in k_range:
    #     error_rate = handwritingTest(test_path='testDigits', train_path='trainingDigits', k=k)
    #     k_error.append(error_rate)
    #
    # # 画图，x轴为k值，y值为误差值
    # plt.plot(k_range, k_error)
    # plt.xlabel("Value of K for KNN")
    # plt.ylabel("Error Rate")
    # plt.show()


    print("测试手写数字".center(40, "*"))
    for i in range(10):
         pic_path = "test_data\{}.jpg".format(str(i)*2)
         testMyHandWriteImgRecoginze(test_pic_path=pic_path, k=3)
