from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (confusion_matrix, classification_report,
                             matthews_corrcoef, precision_score,
                             roc_auc_score, accuracy_score,f1_score,precision_recall_curve,roc_curve)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import seaborn as sns
import palettable
from sklearn.model_selection import train_test_split
seed = 10
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, confusion_matrix,precision_recall_curve, roc_curve, auc, fbeta_score,roc_auc_score
from imblearn.metrics import geometric_mean_score


# 定义评估指标
def evaluate(X, y, estm):
    # Performance metrics

    y_pred = estm.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    #     print(confusion_matrix(y, y_pred).ravel())

    # ROC curve
    try:
        if "decision_function" not in dir(estm):
            y_prob = estm.predict_proba(X)[:, 1]
        else:
            y_prob = estm.decision_function(X)
        pre, rec, _ = precision_recall_curve(y, y_prob)
        fpr, tpr, _ = roc_curve(y, y_prob)
        aucroc = auc(fpr, tpr)
        aucpr = auc(rec, pre)
    except AttributeError:
        print("Classifier don't have predict_proba or decision_function, ignoring roc_curve.")
        pre, rec = None, None
        fpr, tpr = None, None
        aucroc = None
        aucpr = None
    eval_dictionary = {
        "ACC": (tp + tn) / (tp + fp + fn + tn),  # 精确度
        "F1": fbeta_score(y, y_pred, beta=1),
        "F2": fbeta_score(y, y_pred, beta=2),
        "GMean": geometric_mean_score(y, y_pred, average='binary'),  # 几何平均值
        "SEN": tp / (tp + fn),  # 灵敏度
        "PREC": tp / (tp + fp),  # 精度
        "SPEC": tn / (tn + fp),  # 特异性
        "MCC": matthews_corrcoef(y, y_pred),  # 马修斯相关系数
        "AUCROC": aucroc
    }
    return eval_dictionary


def muti_score(estm, df_X, df_y):
    eval_dict = []
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)  # K-折叠交叉验证。
    for train_index, test_index in kf.split(df_X, df_y):
        X_train, X_test = df_X[train_index], df_X[test_index]
        y_train, y_test = df_y[train_index], df_y[test_index]
        estm.fit(X_train, y_train)
        eval_dictionary = evaluate(X_test, y_test, estm)
        eval_dict = eval_dict + [eval_dictionary]

    return eval_dict

X_test = pd.read_csv('results/predict_proba/X_test_pf.csv')
y_test = pd.read_csv('data/y_test.csv')
X_train = pd.read_csv('results/predict_proba/X_train_pf.csv')
y_train = pd.read_csv('data/y_train.csv')

# X_test1 = pd.read_csv('results/predict_proba1/X_test_pf.csv').to_numpy()
# y_test1 = pd.read_csv('data/y_test.csv').to_numpy()
# X_train1 = pd.read_csv('results/predict_proba1/X_train_pf.csv').to_numpy()
# y_train1 = pd.read_csv('data/y_train.csv').to_numpy()
clf2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=seed)
from imblearn.under_sampling import NearMiss
nm = NearMiss(version=3)

X_train_balanced, y_train_balanced = nm.fit_resample(X_train, y_train)
clf2.fit(X_train_balanced, y_train_balanced)

pred_label = clf2.predict(X_test)
prob = clf2.predict_proba(X_test)
# np.save('Voting-ACP500_pred_label.npy', pred_label)
# y_test = y_train
CM = confusion_matrix(y_pred=pred_label, y_true=y_test)
TN, FP, FN, TP = CM.ravel()
mcc = matthews_corrcoef(y_pred=pred_label, y_true=y_test)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = precision_score(y_pred=pred_label, y_true=y_test)
roc = roc_auc_score(y_true=y_test, y_score=prob[:,1])
acc = accuracy_score(y_true=y_test, y_pred=pred_label)
F1 = f1_score(y_true=y_test, y_pred=pred_label)
print('mcc:', mcc)
print('sensitivity:', sensitivity)
print('specificity:', specificity)
print('precision:', precision)
print("GMean",geometric_mean_score(y_true = y_test, y_pred=pred_label, average='binary'))
print('f1',F1)
print('f2',metrics.fbeta_score(y_true=y_test, y_pred=pred_label, beta=2.0))
print('roc:', roc)
print('acc:', acc)