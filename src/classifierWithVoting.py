from sklearn import svm
import csv
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.utils import resample
from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import random
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import KFold

'''
SOFT VOTING:
AVERAGE CORRECCT CLASSIFICATION ON ALL RUNS:  0.965830721003
AVERAGE MAD ON ALL RUNS:  0.379623824451

HARD VOTING:
AVERAGE CORRECCT CLASSIFICATION ON ALL RUNS:  0.966457680251
AVERAGE MAD ON ALL RUNS:  0.452821316614

'''

print "Loading data.."
cnt = 0

ds = []
labels = [ ]

num_folds = 5
num_runs = 20

with open('../data/winequality-red.csv', 'rb') as f:
    reader = csv.reader(f,delimiter = ';')
    for row in reader:
        if cnt == 0:
            cnt = cnt + 1
            continue
        f = float(row[len(row)-1])
        l = [float(x) for x in row[:len(row)-1]]
        ds.append(l)
        labels.append(f)

ds = np.array(ds)
labels = np.array(labels)


CCRSum = 0
MADSum = 0
for idx in range(num_runs):
    kf = KFold (len(ds),n_folds = num_folds, shuffle=True)
    for k, (train, test) in enumerate(kf):
        trainds, testds = ds[train], ds[test]
        trainlabels, testlabels = labels[train], labels[test]

    print "RUN: ", idx
    print "DS: ", len(ds)
    print "Train DS: ", len(trainds)
    print "Test DS: ", len(testds)
    print "Training"

    clfSVM =  svm.SVC(kernel = 'rbf', probability = True, max_iter = 1000, verbose = False)
    clfGrd = GradientBoostingClassifier (n_estimators = 1000, max_depth = 10)
    clfLogReg = LogisticRegression ()




    eclf = VotingClassifier(estimators=[('svm', clfSVM), ('gb', clfGrd), ('log',clfLogReg)], voting='hard', weights = [1,1,3])
    eclf.fit (trainds,trainlabels)


    print "Predicting"

    outputs = eclf.predict (testds)




    print "Testing"

    madCnt = 0
    t = 1
    correct = 0
    for index in range (len(outputs)):
        out = int(outputs[index])
        target = int(testlabels[index])
        if abs(out-target) <=t:
            correct = correct + 1
        madCnt = madCnt + ( abs(target-out) )


    MADSum = MADSum + (float(madCnt)/len(outputs))
    CCRSum = CCRSum + float(correct)/len(outputs)

    print "Correct Classification: "
    print float(correct)/len(outputs)
    print "MAD: "
    print float(madCnt)/len(outputs)
    print "---------------------------------------------"
    print "---------------------------------------------"

print "END"
print "AVERAGE CORRECCT CLASSIFICATION ON ALL RUNS: ", float(CCRSum)/num_runs
print "AVERAGE MAD ON ALL RUNS: ", float(MADSum)/num_runs
