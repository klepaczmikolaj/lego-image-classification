from sklearn.svm import SVC
from numpy import load
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from scipy import interp

feat_train = load('feat_train.npy')
feat_test = load('feat_test.npy')
train_batches = load('train_batches.npy')
test_batches = load('test_batches.npy')

y_true = label_binarize(test_batches, classes=list(range(20)))

for kernel in ['linear', 'rbf', 'poly']:
    for C in [0.01, 0.1, 1, 10, 100, 1000]:
        if kernel == 'poly':
            svm = OneVsRestClassifier(SVC(C = C, kernel=kernel, degree = 2, probability=True, gamma = 'auto', random_state = 1))
        else:
            svm = OneVsRestClassifier(SVC(C = C, kernel=kernel, probability=True, gamma ='auto', random_state = 1))
    
        svm.fit(feat_train,train_batches)

        predict = svm.predict_proba(feat_test)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()


        for i in range(20):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], predict[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(20)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(20):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= 20

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        print(roc_auc["macro"])
        sumAuc = sum(roc_auc.values())

        print('kernel = ' + kernel + ' C = ' + str(C) + ' mean auc = ' + str(sumAuc/20))

        top1 = 0.0
        top5 = 0.0 

        for i, l in enumerate(test_batches):
            class_prob = predict[i]
            top_values = (-class_prob).argsort()[:5]
            if top_values[0] == l:
                top1 += 1.0
            if np.isin(np.array([l]), top_values):
                top5 += 1.0

        print("top1 acc", top1/len(test_batches))
        print("top5 acc", top5/len(test_batches))
