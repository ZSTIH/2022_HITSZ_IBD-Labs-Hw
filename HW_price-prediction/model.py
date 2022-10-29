# -*- coding: UTF-8 -*-
# 为方便测试，请统一使用 numpy、pandas、sklearn 三种包，如果实在有特殊需求，请单独跟助教沟通
import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

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
        df_train = df_train.drop(labels=['出售价格'], axis=1)
        self.category_columns = ['所属区域', '社区', '建筑类型', '当前税收级别', '街区', '地段', '当前建筑类别',
                                 '地址', '公寓号', '邮编', '居住单元', '商业单元', '总单元',
                                 '出售时税收级别', '出售时建筑类别']
        self.numeric_columns = ['土地平方英尺', '总平方英尺', '修建年份', '出售日期']
        self.data_path = "total_data_processed.csv"
        self.X_train = None
        self.X_test = None
        self.df_predict = pd.DataFrame(index=df_test.index)

        # 初始化模型
        self.predict_model = GradientBoostingRegressor(n_estimators=500)
        # 初始化对缺失(异常)数据进行预测的模型
        self.data_predict_model = ExtraTreesRegressor(n_estimators=500, bootstrap=True, n_jobs=-1)
        self.preprocess(df_train, df_test)

    # 模型训练，输出训练集平均绝对误差和均方误差
    def train(self):
        self.predict_model.fit(self.X_train, self.y_train)
        y_train_pred = self.predict_model.predict(self.X_train)
        y_train_pred = [max(0, num) for num in y_train_pred]  # 将小于0的结果置0
        return mean_absolute_error(self.y_train, y_train_pred), mean_squared_error(self.y_train, y_train_pred)

    # 模型测试，输出测试集预测结果
    def predict(self):
        y_test_pred = self.predict_model.predict(self.X_test)
        y_test_pred = [max(0, num) for num in y_test_pred]  # 将小于0的结果置0
        self.df_predict['出售价格'] = y_test_pred
        return self.df_predict

    # 数据预处理
    def preprocess(self, df_train, df_test):
        train_size = df_train.shape[0]
        # 由于数据预处理时间过久，因此将上一次数据预处理的结果存起来便于下一次加载使用
        # 检测到目录下有经过预处理的数据时，直接加载，不必重复进行数据预处理步骤
        if os.path.exists(self.data_path):
            total_data = pd.read_csv(self.data_path)
        else:
            total_data = pd.concat([df_train, df_test], ignore_index=True)
            # 属性“地役权”在全体样本中都缺失了，因此可以直接删去
            total_data = total_data.drop(labels=['地役权'], axis=1)

            # 处理类别型属性(直接编号)
            print("正在处理类别型属性...")
            for feature in self.category_columns:
                value_list = []
                total_data_index = 0
                for value in total_data[feature]:
                    if value not in value_list:
                        value_list.append(value)
                    total_data.loc[total_data_index, feature] = value_list.index(value)
                    total_data_index += 1
                print("类别型属性[%s]被处理完毕" % feature)
            print("全部类别型属性都已被处理完毕！")
            # 处理数值型属性(用类别型属性和正常值进行预测)
            print("正在处理数值型属性...")
            for feature in self.numeric_columns:
                if feature != '出售日期':
                    # 获得数值型属性没有异常的项
                    numeric_valid = total_data.loc[
                        (total_data[feature] != '0') & (total_data[feature] != ' -  ')]
                    numeric_valid_X = numeric_valid[self.category_columns]
                    numeric_valid_y = numeric_valid[feature]
                    self.data_predict_model.fit(numeric_valid_X, numeric_valid_y)
                    total_data_index = 0
                    for value in total_data[feature]:
                        if value == ' -  ' or value == '0':
                            numeric_abnormal = total_data.iloc[total_data_index, :]
                            numeric_abnormal = np.array([numeric_abnormal[self.category_columns]])
                            predict_res = self.data_predict_model.predict(numeric_abnormal)
                            # 使用预测值来填充数值型属性的缺失(异常)值
                            total_data.loc[total_data_index, feature] = int(predict_res[0])
                        total_data_index += 1
                else:
                    total_data_index = 0
                    for value in total_data[feature]:
                        time_array = time.strptime(value, "%Y-%m-%d %H:%M:%S")
                        timestamp = time.mktime(time_array)
                        # 将出售日期对应的字符串转换为对应时间戳
                        total_data.loc[total_data_index, feature] = int(timestamp)
                        total_data_index += 1
                print("数值型属性[%s]被处理完毕" % feature)
            # 将处理后的数据存入csv
            total_data.to_csv(self.data_path, index=False, encoding='utf-8-sig')
            print("数据处理完毕且已经被存入文件%s中！" % self.data_path)
        self.X_train = total_data.iloc[:train_size, :]
        self.X_test = total_data.iloc[train_size:, :]


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
