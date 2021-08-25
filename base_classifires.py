import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import KFold
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, confusion_matrix,precision_recall_curve, roc_curve, auc, fbeta_score,roc_auc_score
import warnings
import os
from fea_extract import read_fasta,insert_PAAC,insert_CTD,insert_QSO
warnings.filterwarnings('ignore')
Path('./results/predict_proba/').mkdir(exist_ok=True,parents=True)
Path('./results/evaluate/').mkdir(exist_ok=True,parents=True)

seed = 10

#DefineBaseLearner
def base_clf(clf, X_train, y_train, model_name, n_folds=7):
    ntrain = X_train.shape[0]  # 训练样本个数。
    nclass = len(np.unique(y_train))  # 类别个数。
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)  # K-折叠交叉验证。
    base_pf_train = np.zeros((ntrain, nclass))  # 返回给定形状和类型的新数组，用零填充。
    base_cf_train = np.zeros((ntrain))

    for train_index, test_index in kf.split(X_train, y_train):
        kf_X_train, kf_y_train = X_train[train_index], y_train[train_index]
        kf_X_test = X_train[test_index]

        clf.fit(kf_X_train, kf_y_train)
        base_pf_train[test_index] = clf.predict_proba(kf_X_test)
        base_cf_train[test_index] = clf.predict(kf_X_test)
    clf.fit(X_train, y_train)
    joblib.dump(model, f'./Models/base/{model_name}')
    return base_pf_train[:, -1], base_cf_train


def evaluate(X, y, estm):
    # Performance metrics
    y_pred = estm.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(confusion_matrix(y, y_pred).ravel())

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
        "AUC": roc_auc_score(y, y_prob)
    }
    return eval_dictionary

XGBoost = XGBClassifier(random_state=seed)
LR  = LogisticRegression(solver='lbfgs', multi_class='multinomial',random_state=seed)

#Exporting training and test sets
train_sets = {
    lab: pd.read_csv('data/train/{:s}_train.csv'.format(lab))
    for lab in ['seq_avp', 'seq_anticov']
}
test_sets = {
    lab: pd.read_csv('data/test/{:s}_test.csv'.format(lab))
    for lab in ['seq_avp', 'seq_anticov']
}
train_sets['seq_anticov'].loc[:, 'Label'] = 1  # loc[]通过行标签索引行数据
train_sets['seq_avp'].loc[:, 'Label'] = 0  # 负数据集
test_sets['seq_anticov'].loc[:, 'Label'] = 1
test_sets['seq_avp'].loc[:, 'Label'] = 0
all_train = pd.concat([train_sets['seq_anticov'], train_sets['seq_avp']], ignore_index='ignore')
all_test = pd.concat([test_sets['seq_anticov'], test_sets['seq_avp']], ignore_index='ignore')
X_train = all_train.iloc[:, 0:2]
X_test = all_test.iloc[:, 0:2]
y_train = all_train['Label']
y_test = all_test["Label"]
X_test.to_csv('data/X_test.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)
X_train.to_csv('data/X_train.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
num = [7,5,0,7,13,9,9,7]
encod = ['CTD','CTD','CTD','QSO','PAAC','QSO','CTD','PAAC']
m = ['XGBoost']
for i in range(len(num)):
    X = pd.read_csv('results/process_data/train_{}_{}.csv'.format(str(num[i]),encod[i]))
    feature = X.columns[2:]
    X_train = X[feature].to_numpy()
    y_train = pd.read_csv('results/process_data/y_train_{}.csv'.format(str(num[i]))).to_numpy()
    for c in m:
        model = eval(c)
        base_clf(model,X_train,y_train,f'{num[i]}_{encod[i]}_{c}.m')


insert_str=["insert_PAAC","insert_CTD","insert_QSO"]

print("Transform data{:s} to features dataframe...".format('X_test'), end=" ")
for j in range(len(insert_str)):
    df_seq = pd.read_csv('data/X_test.csv')
    df_seq1 = eval(insert_str[j])(df_seq)
    df_seq1.to_csv('results/process_data/{}_{}.csv'.format('test',insert_str[j][7:]),index =False )
    print("data{}and{}".format('test',insert_str[j][7:]))

print("Transform data{:s} to features dataframe...".format('X_train'), end=" ")
for j in range(len(insert_str)):
    df_seq = pd.read_csv('data/X_train.csv')
    df_seq1 = eval(insert_str[j])(df_seq)
    df_seq1.to_csv('results/process_data/{}_{}.csv'.format('train',insert_str[j][7:]),index =False )
    print("data{}and{}".format('train',insert_str[j][7:]))
#test dataset
X_test_base_pf = []
X_test_base_cf = []
X_test_base_pcf = []
all_eval_dic = []
Index = []
for i in range(len(num)):
    X = pd.read_csv('results/process_data/test_{}.csv'.format(encod[i]))
    feature = X.columns[2:]
    X_test = X[feature].to_numpy()
    y_test = pd.read_csv('data/y_test.csv').to_numpy()
    for c in m:
        model = joblib.load('Models/base/{}_{}_{}.m'.format(num[i],encod[i],c))
        X_test_pf = model.predict_proba(X_test)[:,-1]
        X_test_cf = model.predict(X_test)
        # PF
        X_test_base_pf.append(X_test_pf)
        X_test_pf_dic = np.array(X_test_base_pf).T
        # CF
        X_test_base_cf.append(X_test_cf)
        X_test_cf_dic = np.array(X_test_base_cf).T
        # PCF
        X_test_base_pcf.append(X_test_pf)
        X_test_base_pcf.append(X_test_cf)
        X_test_pcf_dic = np.array(X_test_base_pcf).T
        # evaluate
        eval_dic = evaluate(X_test,y_test,model)
        print("{}_{}_{}：{}".format(num[i],encod[i],c,eval_dic))
        all_eval_dic = all_eval_dic+[eval_dic]
        Index.append("{}_{}_{}".format(num[i],encod[i],c))
#评估指标结果输出
all_eval_dic = pd.DataFrame(all_eval_dic,index=Index)
all_eval_dic.to_csv(os.path.join("results/evaluate/baselearners/test_evalu.csv"))
#输出概率向量
#训练集
X_test_pf_dic = pd.DataFrame(X_test_pf_dic)
X_test_pf_dic.to_csv(os.path.join("results/predict_proba/X_test_pf.csv"),index=False)
X_test_cf_dic = pd.DataFrame(X_test_cf_dic)
X_test_cf_dic.to_csv(os.path.join("results/predict_proba/X_test_cf.csv"),index=False)
X_test_pcf_dic = pd.DataFrame(X_test_pcf_dic)
X_test_pcf_dic.to_csv(os.path.join("results/predict_proba/X_test_pcf.csv"),index=False)

#train dataset
X_train_base_pf = []
X_train_base_cf = []
X_train_base_pcf = []
all_eval_dic = []
Index = []
eval_d1 = pd.DataFrame()
for i in range(len(num)):
    X = pd.read_csv('results/process_data/train_{}.csv'.format(encod[i]))
    feature = X.columns[2:]
    X_train = X[feature].to_numpy()
    y_train = pd.read_csv('data/y_train.csv').to_numpy()
    for c in m:
        model = joblib.load('Models/base/{}_{}_{}.m'.format(num[i], encod[i], c))
        X_train_pf = model.predict_proba(X_train)[:,-1]
        X_train_cf = model.predict(X_train)
                    # PF
        X_train_base_pf.append(X_train_pf)
        X_train_pf_dic = np.array(X_train_base_pf).T
                    # CF
        X_train_base_cf.append(X_train_cf)
        X_train_cf_dic = np.array(X_train_base_cf).T
                    # PCF
        X_train_base_pcf.append(X_train_pf)
        X_train_base_pcf.append(X_train_cf)
        X_train_pcf_dic = np.array(X_train_base_pcf).T
            # evaluate
        eval_dic = evaluate(X_train, y_train, model)

            #             eval_dict = muti_score(model,X_train1,y_train1)
            #             eval_d = pd.DataFrame(eval_dict).mean()
            #             eval_d = pd.DataFrame(eval_d)
            #             eval_d1 = pd.concat([eval_d1,eval_d],axis = 1)
        print("{}_{}_{}：{}".format(num[i], encod[i], c, eval_dic))
        all_eval_dic = all_eval_dic + [eval_dic]
        Index.append("{}_{}_{}".format(num[i], encod[i], c))
# #评估指标结果输出
all_eval_dic = pd.DataFrame(all_eval_dic, index=Index)
# eval_d1.columns = Index
all_eval_dic.to_csv(os.path.join("results/evaluate/baselearners/train_evalu.csv"))
# #输出概率向量
# #训练集
X_train_pf_dic = pd.DataFrame(X_train_pf_dic)
X_train_pf_dic.to_csv(os.path.join("results/predict_proba/X_train_pf.csv"),index=False)
X_train_cf_dic = pd.DataFrame(X_train_cf_dic)
X_train_cf_dic.to_csv(os.path.join("results/predict_proba/X_train_cf.csv"),index=False)
X_train_pcf_dic = pd.DataFrame(X_train_pcf_dic)
X_train_pcf_dic.to_csv(os.path.join("results/predict_proba/X_train_pcf.csv"),index=False)