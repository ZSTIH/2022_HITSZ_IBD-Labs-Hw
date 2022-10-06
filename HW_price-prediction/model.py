# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import argparse

# 设定随机数种子，保证代码结果可复现
np.random.seed(1024)


class Model:
    """
    要求：
        1. 需要有__init__、train、predict三个方法，且方法的参数应与此样例相同
        2. 需要有self.X_train、self.y_train、self.X_test三个实例变量，请注意大小写
        3. 如果划分出验证集，请将实例变量命名为self.X_valid、self.y_valid
    """
    # 模型初始化，数据预处理，仅为示例
    def __init__(self, train_path, test_path):
        df_train = pd.read_csv(train_path, encoding='utf-8')
        df_test = pd.read_csv(test_path, encoding='utf-8')
        self.y_train = df_train['出售价格'].values
        
        # 简单处理
        self.X_train = pd.concat([df_train['街区'], df_train['地段'], df_train['修建年份']], axis=1)
        self.X_test = pd.concat([df_test['街区'], df_test['地段'], df_test['修建年份']], axis=1)

        # 初始化模型
        self.regression_model = LinearRegression()
        self.df_predict = pd.DataFrame(index=df_test.index)

    # 模型训练，输出训练集平均绝对误差和均方误差
    def train(self):
        self.regression_model.fit(self.X_train, self.y_train)
        y_train_pred = self.regression_model.predict(self.X_train)
        return mean_absolute_error(self.y_train, y_train_pred), mean_squared_error(self.y_train, y_train_pred)

    # 模型测试，输出测试集预测结果
    def predict(self):
        y_test_pred = self.regression_model.predict(self.X_test)
        self.df_predict['出售价格'] = y_test_pred
        return self.df_predict

    # 数据预处理
    def preprocess(self):
        pass


# 以下部分请勿改动！
if __name__ == '__main__':
    # 解析输入参数。在终端执行以下语句即可运行此代码： python model.py --train_path "./data/train.csv" --test_path "./data/test.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="train.csv", help="path to train dataset")
    parser.add_argument("--test_path", type=str, default="test.csv", help="path to test dataset")
    opt = parser.parse_args()

    model = Model(opt.train_path, opt.test_path)

    print(f'训练集维度:{model.X_train.shape}\n测试集维度:{model.X_test.shape}')
    mae_score, mse_score = model.train()
    print(f'mae_score={mae_score:.6f}, mse_score={mse_score:.6f}')
    predict = model.predict()
    predict.to_csv('学号_姓名_submit_n.csv', index=False, encoding='utf-8-sig')
