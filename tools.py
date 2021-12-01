import numpy as np
from Bio import SeqIO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef, confusion_matrix,precision_recall_curve, roc_curve, auc, fbeta_score,roc_auc_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from imblearn.metrics import geometric_mean_score
from fea_extract import read_fasta,insert_AAC,insert_DPC,insert_CKSAAGP,insert_CTD,insert_PAAC,insert_AAI,insert_GTPC,insert_QSO,insert_AAE,insert_ASDC

from sklearn.metrics import precision_recall_curve, average_precision_score

int1D2wordDict = {1: 'G',2: 'A',3: 'V',4: 'L', 5: 'I',6: 'P',7: 'F', 8: 'Y',
                  9: 'W', 10: 'S', 11: 'T', 12: 'C', 13: 'M', 14: 'N', 15: 'Q',
                  16: 'D', 17: 'E', 18: 'K', 19: 'R', 20: 'H', 21: 'X', 22: 'B',
                  23: 'J', 24: 'O', 25: 'U', 26: 'Z'}
# 定义函数
# 读取文件,删除长度小于6大于100的样本

#读取文件
def read_fasta(fname):
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df

def readFile(inFile, spclen):
    spclen = 100
    names = []
    seqs = []
    dict_seqs = {}
    for line in open(inFile):
        if line.startswith('>'):
            name = line.replace('>', ' ').split()[0]
            names.append(name)
            dict_seqs[name] = ''
        else:
            seq = line.replace('\n', '')
            dict_seqs[name] += seq

    for name in names:
        if 6 <= len(dict_seqs[name]) <= spclen:
            dict_seqs[name] = dict_seqs[name] + 'X' * (spclen - len(dict_seqs[name]))
            seqs.append(dict_seqs[name])
        else:
            dict_seqs[name] = dict_seqs[name][0:spcLen]
            seqs.append(dict_seqs[name])
    np_seq = np.vstack([names, seqs]).T
    return dict_seqs, names, seqs


# 定义函数
def del_data(inFile):
    seq = read_fasta(inFile)
    seqname = seq.to_numpy()
    newseq = []
    j = seqname.shape[0]
    for i in range(j):
        if 6 <= len(seqname[i][1])<=100:
            newseq.append(seqname[i])
    newseq = np.array(newseq)
    print('序列去除小于6序列后的维度：', newseq.shape)
    newseq = pd.DataFrame(data=newseq, columns=["Id", "Sequence"])
    return newseq


# 填充序列
def padseq(seqname):
    seq = seqname.to_numpy()
    newseq = []
    j = seq.shape[0]
    for i in range(j):
        seq[i][1] = seq[i][1] + 'X' * (100 - len(seq[i][1]))
        newseq.append(seq[i])
    newseq = np.array(newseq)
    newseq = pd.DataFrame(data=newseq, columns=["Id", "Sequence"])
    return newseq

#模型评价
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
        "GMean": geometric_mean_score(y, y_pred, average='binary'), #几何平均值
        "SEN": tp / (tp + fn), #灵敏度
        "PREC": tp / (tp + fp), #精度
        "SPEC": tn / (tn + fp), #特异性
        "MCC": matthews_corrcoef(y, y_pred), #马修斯相关系数
        "AUC": roc_auc_score(y, y_prob),
        "AUPR":average_precision_score(y, y_prob)
    }
    return eval_dictionary

def cv(model,df_X,df_y,n_folds=5):
    eval_dict = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=10)#K-折叠交叉验证。
#     model1 = model
    for train_index, test_index in kf.split(df_X, df_y):
        X_train,X_test=df_X[train_index],df_X[test_index]
        y_train,y_test=df_y[train_index],df_y[test_index]
        model.fit(X_train,y_train)
        eval_dictionary = evaluate(X_test,y_test,model)
        eval_dict = eval_dict+[eval_dictionary]
    evals = pd.DataFrame(eval_dict).mean()
    Evals = pd.DataFrame(evals)
    return Evals


def Int2word(data):
    int_encoding = []
    for seq in data:
        encoding = []
        for i in range(100):
            encoding.append(int1D2wordDict[seq[i]])
        int_encoding.append(encoding)
    int_encodings = pd.DataFrame(int_encoding).to_numpy()
    Seq = []
    for s in int_encodings:
        s = filter(lambda m: m != 'X', s)
        s= list(s)
        Seq.append(s)
    encodings = np.array(Seq)
    enco = []
    for i in encodings:
        seq = ''.join(i)
        enco.append(seq)
    return enco  

def pro_data(seq):
    df_n = insert_PAAC(seq)
    df_n = insert_AAC(df_n)
    df_n = insert_CKSAAGP(df_n)
    df_n = insert_CTD(df_n)
    df_n = insert_DPC(df_n)
    df_n = insert_GTPC(df_n)
    df_n = insert_QSO(df_n)
    df_n = insert_AAE(df_n)
    df_n = insert_ASDC(df_n)
    return df_n