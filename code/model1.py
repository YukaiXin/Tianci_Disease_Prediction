import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error

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
for i in train.columns:
    if i not in del_feat:
        feat.append(i)
train=train[feat]
feat.remove('血糖') #测试集不需要‘血糖’属性
test=test[feat]

y_train = train["血糖"]
x_train = train.drop(['id','血糖'], axis=1)
y_test = test.drop('id', axis=1)

# training xgboost
dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(y_test)

params = {'booster': 'gbtree',
          'objective': 'reg:linear',
          'eval_metric': 'rmse',
          'max_depth': 6,#通常取值：3-10
          'gamma':0.2,#给定了所需的最低loss function的值
          'lambda': 100,
          'subsample': 1,#用于训练模型的子样本占整个样本集合的比例
          'colsample_bytree': 0.6,
          'min_child_weight': 12,  # 5~10,孩子节点中最小的样本权重和，即调大这个参数能够控制过拟合
          'eta': 0.02,#更新过程中用到的收缩步长，取值范围为：[0,1]
          'sample_type': 'uniform',
          'normalize': 'tree',
          'rate_drop': 0.1,
          'skip_drop': 0.9,
          'seed': 100,
          'nthread':-1
          }

bst_nb = 700
watchlist = [(dtrain, '训练误差')]
model = xgb.train(params, dtrain, num_boost_round=bst_nb, evals=watchlist)  # 训练模型

y_pred = model.predict(dtest)

# print((mean_squared_error(validate,y_pred))/2)
y_predDF=pd.DataFrame({None:y_pred})
y_predDF.to_csv("SMUDMers_test_B_res.csv",header=None,index=False,float_format="%.2f")