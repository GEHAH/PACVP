from sklearn.model_selection import train_test_split,KFold
import pandas as pd
import os, re, math, platform
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from fea_extract import read_fasta,insert_PAAC,insert_CTD,insert_QSO
seed = 10
from pathlib import Path
Path('./results/process_data/').mkdir(exist_ok=True,parents=True)

#Define function
#Remove sequences of length less than 6 and greater than 100
def del_data(Filename):
    seqname = read_fasta(Filename)
    seqname = seqname.to_numpy()
    newseq = []
    j = seqname.shape[0]
    for i in range(j):
        if 6<=len(seqname[i][1])<=100:
            newseq.append(seqname[i])
    newseq = np.array(newseq)
    print('Dimension after sequence removal of less than 6 sequences:',newseq.shape)
    newseq = pd.DataFrame(data = newseq,columns=["Id", "Sequence"])
    return newseq
#Delineate the data set
def splits_dataset(X, n_folds):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    s=0
    for train_index, test_index in kf.split(X):
        kf_X_test = X.iloc[test_index]
        ss=str(s)
        if len(kf_X_test)> len(X)/n_folds:
            test_index1 = test_index.tolist()
            kf_X_test.drop([test_index1[-1]],inplace=True)
        kf_X_test.to_csv(os.path.join( "data/split_data/seq_avp_train_{:s}.csv".format(ss)), index=False)
        s=s+1

insert_str=["insert_PAAC","insert_CTD","insert_QSO"]
def splitdata_15(Filename):
    data = pd.read_csv(Filename)
    splits_dataset(data,n_folds=15)

#main
def process_data(filename1,filename2):
    seq_avp = del_data(filename1)
    seq_anticov = del_data(filename2)
    prefix = [seq_avp, seq_anticov]
    Prefix = ['seq_avp', 'seq_anticov']
    for i in range(2):
        df_train, df_test = train_test_split(prefix[i], random_state=seed, test_size=.3)
        df_train.to_csv(os.path.join("data/train/{:s}_train.csv".format(Prefix[i])), index=False)
        df_test.to_csv(os.path.join("data/test/{:s}_test.csv".format(Prefix[i])), index=False)
        print("Done!")
    splitdata_15('data/train/seq_avp_train.csv')
    for i in range(15):
        train_p = pd.read_csv('data/train/seq_anticov_train.csv')
        train_n = pd.read_csv('data/split_data/seq_avp_train_{:s}.csv'.format(str(i)))
        train_p.loc[:, 'Label'] = 1
        train_n.loc[:, 'Label'] = 0
        all_train = pd.concat([train_p, train_n], ignore_index='ignore')
        X_train = all_train.iloc[:, 0:2]
        y_train = all_train['Label']
        X_train.to_csv('results/process_data/X_train_{}.csv'.format(str(i)), index=False)
        y_train.to_csv('results/process_data/y_train_{}.csv'.format(str(i)), index=False)
        for j in range(len(insert_str)):
            print("data{}_{}".format(str(i), insert_str[j][7:]))
            X_train = pd.read_csv('results/process_data/X_train_{}.csv'.format(str(i)))
            df_seq = eval(insert_str[j])(X_train)
            df_seq.to_csv('results/process_data/train_{}_{}.csv'.format(str(i), insert_str[j][7:]), index=False)


if __name__ == '__main__':
    process_data('data/Anti-Virus.faa','data/Anti-CoV.csv')

