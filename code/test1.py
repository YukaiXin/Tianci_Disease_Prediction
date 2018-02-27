
import pandas as pd
# 将两部分的训练集train1，train2共同组合成总得训练集train
train1=pd.read_csv(r"../dataset/d_train_20180102.csv",encoding='gbk')

print(train1.isnull().sum())
print("ok !!!")