import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.svm as svm
import pandas as pd
import copy
import time
import random
from operator import itemgetter
import sklearn.ensemble as ensemble
import scipy.optimize as optimize
from sklearn.linear_model import LogisticRegression
from sympy import *

def arff2csv(filename, compression = False):
    f = open("../dataset/{}".format(filename), "r")
    header = []
    data = []
    sub_data = []
    for (i,row) in enumerate(f):
        row_list = row.split(" ")
        if row_list[0] == "@attribute":
            header.append(row_list[1])
        elif row_list[0] != "@relation" and row != "\n" and row != "@data\n":
            if compression:
                sub_data.append(row.strip("{").strip("}\n").split(","))
            else:
                data.append(row.strip("\n").split(","))

    if compression:
        attribute = len(header)
        for row in sub_data:
            d = ['0' for i in range(attribute)]
            for r in row:
                d[int(r.split(" ")[0])] = str(r.split(" ")[1])
            data.append(d)
    output_filename = filename.split(".")[0]
    output_file = open("../csv/{}.csv".format(output_filename.split("/")[-1]), "w")
    output_file.write(",".join(header))
    output_file.write("\n")
    for row in data:
        output_file.write(",".join(row))
        output_file.write("\n")

def read_csv2df(filename):
    return pd.read_csv(filename)

class BinaryRelevance:
    def __init__(self,classifier):
        self.classifier = classifier
        self.model_list = []

    def fit(self, train_X, train_S):
        train_S.shape[1]
        for i in range(train_S.shape[1]):
            clf = copy.deepcopy(self.classifier)
            clf.fit(train_X, train_S[:,i])
            self.model_list.append(clf)

    def predict(self, test_X):
        pred_S_list = []
        for clf in self.model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
        pred_S = np.array(pred_S_list)
        return pred_S.T

class ClassifierChains:
    """ The Classifier Chains Model : Classifier Chains for Multi-label Classification"""
    def __init__(self,classifier):
        self.classifier = classifier
        self.model_list = []
        self.order = []

    # CCの訓練 x: 属性ベクトルの集合, S: 解答ベクトルの集合, order: chainの順番をlistにしたもの
    def fit(self, train_X, train_S, order = False):
        if order == False:
            order = [i for i in range(train_S.shape[1])]
        self.order = order
        # train_xとtrain_Sの型がdataflame型ならばnumpy型に変更すべき
        for i in order:
            clf = copy.deepcopy(self.classifier)
            clf.fit(train_X, train_S[:,i])
            self.model_list.append(clf)
            train_X = np.c_[train_X, train_S[:,i]]

    def predict(self, test_X):
        pred_S_list = []
        for clf in self.model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
            # この予想できたpred_yをtest_xの右側に結合するために縦ベクトルに変換。
            pred_Y = np.matrix(pred_Y).T
            test_X = np.c_[test_X, pred_Y]
        shape = np.array(pred_S_list).shape
        pred_S = np.zeros(shape)
        # 答えを並び替える。
        for (i, pred) in zip(self.order, pred_S_list):
            pred_S[i] = np.array(pred)
        return pred_S.T

class EnsembleClassifierChains:
    """ Ensemble of the Classifier Chains Model : Classifier Chains for Multi-label Classification"""
    def __init__(self, classifier):
        self.classifier = classifier
        self.__model_list = []
        self.__order = []
        self.__n = 3

    # CCの訓練 x: 属性ベクトルの集合, S: 解答ベクトルの集合, order: chainの順番をlistにしたもの
    def __CC_train(self, train_X, train_S, order):
        model_list = []
        # train_xとtrain_Sの型がdataflame型ならばnumpy型に変更すべき
        for i in order:
            clf = copy.deepcopy(self.classifier)
            clf.fit(train_X, train_S[:,i])
            model_list.append(clf)
            train_X = np.c_[train_X, train_S[:,i]]
        return model_list

    def __make_order_list(self, num_of_labels, n):
        order = [list(range(num_of_labels)) for i in range(n)]
        for ls in order:
            random.shuffle(ls)
        return np.array(order)

    def __random_sampling(self, train_X, train_S, rate):
        if 0 < rate <= 1:
            X_size = train_X.shape[1]
            S_size = train_S.shape[1]
            train_data = np.c_[train_X, train_S]
            np.random.shuffle(train_data)
            data_size = int(train_data.shape[0] * rate)
            train_data = train_data[:data_size]
            return train_data[:,:X_size], train_data[:,X_size:]
        else:
            sys.exit("ERROR: 0 < rate <= 1 ")


    # n is roop count.
    def fit(self, train_X, train_S, n, rate):
        # shuffleされたlistがnumpy.array形式で返ってくる。
        self.__order = self.__make_order_list(train_S.shape[1],n)
        self.__n = n
        # trainデータの67%を用いてn回学習する。
        for i in range(n):
            # ここでtrain_small_X, train_small_Sを作成する必要がある。
            train_small_X, train_small_S = self.__random_sampling(train_X, train_S, rate)
            self.__model_list.append(self.__CC_train(train_small_X, train_small_S, self.__order[i]))

    def __CC_predict(self, test_X, order, model_list):
        pred_S_list = []
        for clf in model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
            # この予想できたpred_yをtest_xの右側に結合するために縦ベクトルに変換。
            pred_Y = np.matrix(pred_Y).T
            test_X = np.c_[test_X, pred_Y]
        shape = np.array(pred_S_list).shape
        pred_S = np.zeros(shape)
        # 答えを並び替える。
        for (i, pred) in zip(order, pred_S_list):
            pred_S[i] = np.array(pred)
        return pred_S.T

    def predict(self, test_X, threshold):
        for (i, models) in enumerate(self.__model_list):
            if i == 0:
                pred_S = self.__CC_predict(test_X, self.__order[i], models)
            else:
                pred_S += self.__CC_predict(test_X, self.__order[i], models)
        pred = pred_S / self.__n
        pred_S = np.zeros(pred.shape)
        pred_S[pred > threshold] = 1
        return pred_S

class EnsembleBinaryRelevance:
    def __init__(self,classifier):
        self.classifier = classifier
        self.__model_list = []
        self.__n = 3

    def __BR_train(self, train_X, train_S):
        model_list = []
        for i in range(train_S.shape[1]):
            clf = copy.deepcopy(self.classifier)
            clf.fit(train_X, train_S[:,i])
            model_list.append(clf)
        return model_list

    def __BR_predict(self, test_X, model_list):
        pred_S_list = []
        for clf in model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
        pred_S = np.array(pred_S_list)
        return pred_S.T

    def __random_sampling(self, train_X, train_S, rate):
        if 0 < rate <= 1:
            X_size = train_X.shape[1]
            S_size = train_S.shape[1]
            train_data = np.c_[train_X, train_S]
            np.random.shuffle(train_data)
            data_size = int(train_data.shape[0] * rate)
            train_data = train_data[:data_size]
            return train_data[:,:X_size], train_data[:,X_size:]
        else:
            sys.exit("ERROR: 0 < rate <= 1 ")


    # n is roop count.
    def fit(self, train_X, train_S, n, rate):
        # shuffleされたlistがnumpy.array形式で返ってくる。
        self.__n = n
        # trainデータの67%を用いてn回学習する。
        for i in range(n):
            # ここでtrain_small_X, train_small_Sを作成する必要がある。
            train_small_X, train_small_S = self.__random_sampling(train_X, train_S, rate)
            self.__model_list.append(self.__BR_train(train_small_X, train_small_S))

    def predict(self, test_X, threshold):
        for (i, models) in enumerate(self.__model_list):
            if i == 0:
                pred_S = self.__BR_predict(test_X, models)
            else:
                pred_S += self.__BR_predict(test_X, models)
        pred = pred_S / self.__n
        pred_S = np.zeros(pred.shape)
        pred_S[pred > threshold] = 1
        return pred_S

class multi_label_evaluation_metric:
    def __init__(self):
        pass

    def __union_value(self,s, y):
        return ((s + y) - s * y).sum()

    def __intersection_value(self, s, y):
        return (s * y).sum()

    # S is the actual label set. Y is the predicted label set.
    # S and Y are numpy array.
    def accuracy(self, S, Y):
        eval_value = 0
        if S.shape != Y.shape:
            sys.exit("ERROR: The shapes of S and Y are different!")
        for (Si, Yi) in zip(S, Y):
            eval_value += self.__intersection_value(Si,Yi) / self.__union_value(Si,Yi)
        return eval_value / len(S)

    def F_measure_macro(self, S, Y):
        eval_value = 0
        for (Sj, Yj) in zip(S.T,Y.T):
            Sj = Sj.astype(np.bool)
            Yj = Yj.astype(np.bool)
            TPj = (Sj & Yj).astype(np.int).sum()
            FPj = (~Sj & Yj).astype(np.int).sum()
            FNj = (Sj & ~Yj).astype(np.int).sum()
            TNj = (~Sj & ~Yj).astype(np.int).sum()
            eval_value += (2 * TPj) / (2 * TPj + FNj + FPj)
        eval_value = eval_value / S.shape[1]
        return eval_value

def LCard(S):
    return sum(Si.sum() for Si in S) / S.shape[0]

def PDist(S):
    LP_S = []
    for (i,train) in enumerate(S):
        num = 0
        for (j,t) in enumerate(train):
            num += (2 ** j) * t
        LP_S.append(num)
    return len(set(LP_S)) / S.shape[0]

def multi_labeled_ness_info(S):
    print("LCard : {}".format(LCard(S)))
    print("PDist : {}".format(PDist(S)))

class MetaStacking:
    def __init__(self,classifier):
        self.__classifier = classifier
        self.__first_model_list = []
        self.__second_model_list = []

    def __first_stage_train(self, train_X, train_S):
        model_list = []
        for i in range(train_S.shape[1]):
            clf = copy.deepcopy(self.__classifier)
            clf.fit(train_X, train_S[:,i])
            model_list.append(clf)
        return model_list

    def __first_stage_predict(self, test_X):
        pred_S_list = []
        for clf in self.__first_model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
        pred_S = np.array(pred_S_list)
        return pred_S.T # ここなんか論文に 1 - fって書いてあるからそうしている。

    def __second_stage_train(self, train_big_X, train_S):
        model_list = []
        for i in range(train_S.shape[1]):
            clf = copy.deepcopy(self.__classifier)
            clf.fit(train_big_X, train_S[:,i])
            model_list.append(clf)
        return model_list


    def fit(self, train_X, train_S):
        self.__first_model_list = self.__first_stage_train(train_X, train_S)
        added_feature = self.__first_stage_predict(train_X)
        train_big_X = np.c_[train_X, added_feature]
        self.__second_model_list = self.__second_stage_train(train_big_X, train_S)

    def predict(self,test_X):
        pred_S_list = []
        for clf in self.__first_model_list:
            pred_Y = clf.predict(test_X)[:]
            pred_S_list.append(list(pred_Y))
        pred_S = np.array(pred_S_list)
        added_feature = pred_S.T
        test_big_X = np.c_[test_X, added_feature]
        pred_S_list = []
        for clf in self.__second_model_list:
            pred_Y = clf.predict(test_big_X)[:]
            pred_S_list.append(list(pred_Y))
        pred_S = np.array(pred_S_list)
        return pred_S.T

class LabelPowerSets:
    def __init__(self,classifier):
        self.__classifier = classifier
        self.__model = []
        self.__class_num = 0


    # 左から2^0,2^1,2^2,...として足しあわせている。
    def __LP_make(self, train_S):
        LP_S = []
        self.__class_num = train_S.shape[1]
        for (i,train) in enumerate(train_S):
            num = 0
            for (j,t) in enumerate(train):
                num += (2 ** j) * t
            LP_S.append(num)
        return np.array(LP_S)

    def fit(self, train_X, train_S):
        LP_S = self.__LP_make(train_S)
        clf = copy.deepcopy(self.__classifier)
        clf.fit(train_X, LP_S)
        self.__model.append(clf)

    def predict(self, test_X):
        clf = self.__model[0]
        pred_S = []
        pred_Y = clf.predict(test_X)[:]
        for binary in pred_Y:
            pred_S.append([int(value) for value in bin(binary + 2 ** self.__class_num)[:2:-1]])
        return np.array(pred_S)

def draw_heatmap(data, row_labels, column_labels, graph_name):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('{}.png'.format(graph_name))
    return heatmap

def ML_cross_validation(filename, n_labels, n_split, clf_method, met, seed):
    """
    filename: 入力するcsvファイル
    n_labels: ラベルの数
    split_size: 分割数
    """
    df = pd.read_csv(filename).sample(frac=1,random_state=seed)
    X = np.array(df.ix[:,:-n_labels])
    L = np.array(df.ix[:,-n_labels:])

    size = df.shape[0] // n_split
    remainder = df.shape[0] % n_split
    count = 0
    start = 0
    split_range = np.zeros([n_split, 2])
    e_values = []
    e_times = []

    for i in range(n_split):
        if count < remainder:
            split_range[i,0] = int(start)
            split_range[i,1] = int(start + size + 1)
            start += size+1
            count += 1
        else:
            split_range[i,0] = int(start)
            split_range[i,1] = int(start + size)
            start+=size

    for (s,e) in split_range:
        clf = copy.deepcopy(clf_method)
        s = int(s)
        e = int(e)
        train_X = np.delete(X.copy(),list(range(s,e)),0)
        train_L = np.delete(L.copy(),list(range(s,e)),0)
        test_X = X[s:e].copy()
        test_L = L[s:e].copy()

        s_time = time.time()
        clf.fit(train_X, train_L)
        pred = clf.predict(test_X)
        elapsed_time = time.time() - s_time
        #print(pred.shape)
        #print(test_L.shape)
        print(met(test_L, pred))

        e_values.append(met(test_L, pred))
        e_times.append(elapsed_time)
    return np.array(e_values), np.array(e_times)

def main():
    pass

if __name__ == '__main__':
    main()
