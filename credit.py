import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
df  = pd.read_csv(r"D:\kaggle\data\archive\creditcard.csv")
new_df  =df.copy()
new_df['Amount']  = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1,1))

time = new_df['Time']
new_df['Time'] = (time - time.min())/(time.max()-time.min()) 
#非常标准的标准化处理：最小-最大归一化（Min-Max Normalization）
#转换后：
#           （1）'Time'列中的最小值将变为0，最大值将变为1，其他所有值都在0到1之间。
#            （2）这种转换保持了原始数据的相对顺序和分布特征，但改变了数据的尺度。
#优点有：  （1）消除了不同特征之间的尺度差异。
#         （2）有助于加速梯度下降等优化算法的收敛。
#         （3）对于某些对特征尺度敏感的算法（如k-近邻、支持向量机等），可以提高模型性能。
#缺点有：  （1）如果数据集中存在异常值，转换后可能会导致数据分布的改变。
#         （2）转换后的数据可能会对某些模型产生负面影响，特别是在需要保持原始数据分布的情况下。
new_df = new_df.sample(frac=1,random_state=1)
#随机打乱df中的所有行
#种操作通常用于数据预处理阶段，特别是在准备训练机器学习模型之前。
#随机打乱数据可以帮助消除数据中可能存在的顺序偏差，确保后续的训练-验证集分割更加随机和公平
#下面的代码可以起相同的作用
#from sklearn.utils import shuffle
#new_df = shuffle(new_df,random_state=1)
"""
如果只是为了后续的训练-验证集分割，可以直接使用 sklearn.model_selection 中的 
train_test_split 函数，它会自动打乱数据：

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=1)
"""
train , test ,val =  new_df[:240000], new_df[262000:],new_df[262000:]
train_np ,test_np,val_np = train.to_numpy(),test.to_numpy(),val.to_numpy()
x_train ,y_train  = train_np[:,:-1] ,train_np[:,-1]
x_test ,y_test  = test_np[:,:-1] ,test_np[:,-1]
x_val ,y_val  = val_np[:,:-1] ,val_np[:,-1]

rf = RandomForestClassifier(max_depth=5,n_jobs = -1)
rf.fit(x_train,y_train)
print(classification_report(y_val,rf.predict(x_val),target_names=['Not Fraud','Fraud']))
bst  = XGBClassifier(max_depth=5,learning_rate=1,objective='binary:logistic')
bst.fit(x_train,y_train)
print(classification_report(y_val,bst.predict(x_val),target_names=['Not Fraud','Fraud']))