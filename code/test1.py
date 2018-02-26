

print("ok!!!!")
import pandas as pd
# 将两部分的训练集train1，train2共同组合成总得训练集train
train1=pd.read_csv(r"C:/Users/yuki_cool/MyCodes/Tianci_Disease_Prediction/dataset/d_train_20180102.csv",encoding='gbk')
# 合并训练集
train2_1=pd.read_csv(r"C:/Users/yuki_cool/MyCodes/Tianci_Disease_Prediction/dataset/d_test_A_20180102.csv",encoding='gbk')
train2_2=pd.read_csv(r"C:/Users/yuki_cool/MyCodes/Tianci_Disease_Prediction/dataset/d_answer_a_20180128.csv",encoding="gbk",header=None)
train2_2.rename(columns={0:'血糖'},inplace=True) #取名“血糖”
train2=pd.concat([train2_1,train2_2],axis=1)
train=pd.concat([train1,train2],axis=0)


# 删除特别大的‘血糖’异常值
columns=len(train.columns)
train.drop(train.index[[i for i in train.index if train.iloc[i,columns-1]>30]],inplace=True)
# 测试集
test=pd.read_csv(r"C:/Users/yuki_cool/MyCodes/Tianci_Disease_Prediction/dataset/d_test_B_20180128.csv",encoding='gbk')
# validate=pd.read_csv(r"data/d_answer_b_20180130.csv",encoding='utf-8',header=None)
del_feat=['性别','体检日期','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体']
# 删除特征
feat=[]