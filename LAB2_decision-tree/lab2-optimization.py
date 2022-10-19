import random

import numpy as np
import pandas as pd

columns = ['age', 'workclass', 'fnlwgt', 'education', 'educationNum', 'maritalStatus', 'occupation', 'relationship',
           'race', 'sex',
           'capitalGain', 'capitalLoss', 'hoursPerWeek', 'nativeCountry', 'income']
df_train_set = pd.read_csv('./adult.data', names=columns)
df_test_set = pd.read_csv('./adult.test', names=columns, skiprows=1)  # 第一行是非法数据
df_train_set.to_csv('./train_adult.csv', index=False)
df_test_set.to_csv('./test_adult.csv', index=False)

# 数据预处理
# fnlwgt列用处不大，educationNum与education类似
df_train_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)
df_test_set.drop(['fnlwgt', 'educationNum'], axis=1, inplace=True)
df_train_set.drop_duplicates(inplace=True)  # 去除重复行
df_train_set.dropna(inplace=True)  # 去除空行
# 删除有异常值的行
new_columns = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
               'nativeCountry', 'income']
for col in new_columns:
    df_train_set = df_train_set[~df_train_set[col].str.contains(r'\?', regex=True)]
df_train_set.head()

# 处理连续型属性
continuous_column = ['age', 'capitalGain', 'capitalLoss', 'hoursPerWeek']
bins = [0, 25, 50, 75, 100]  # 分箱区间左开右闭 (0, 25], (25, 50], ...
df_train_set['age'] = pd.cut(df_train_set['age'], bins, labels=False)
df_test_set['age'] = pd.cut(df_test_set['age'], bins, labels=False)

# 处理离散型属性
discrete_column = ['workclass', 'education', 'maritalStatus', 'occupation', 'relationship', 'race', 'sex',
                   'nativeCountry', 'income']
workclass_mapping = {' Private': 0, ' Self-emp-not-inc': 1, ' Self-emp-inc': 1, ' Local-gov': 2,
                     ' State-gov': 2, ' Federal-gov': 2, ' Without-pay': 3, ' Never-worked': 3}
df_train_set['workclass'] = df_train_set['workclass'].map(workclass_mapping)
df_test_set['workclass'] = df_test_set['workclass'].map(workclass_mapping)
# 对训练集、测试集同时处理离散型属性education
education_mapping = {' Preschool': 0,
                     ' 1st-4th': 1,
                     ' 5th-6th': 1,
                     ' 7th-8th': 2,
                     ' 9th': 2,
                     ' 10th': 3,
                     ' 11th': 3,
                     ' 12th': 3,
                     ' HS-grad': 3,
                     ' Some-college': 4,
                     ' Bachelors': 5,
                     ' Prof-school': 6,
                     ' Assoc-acdm': 7,
                     ' Assoc-voc': 8,
                     ' Masters': 9,
                     ' Doctorate': 10
                     }
df_train_set['education'] = df_train_set['education'].map(education_mapping)
df_test_set['education'] = df_test_set['education'].map(education_mapping)
# 对训练集、测试集同时处理离散型属性marital-status
marital_status_mapping = {' Married-civ-spouse': 0,
                          ' Divorced': 1,
                          ' Never-married': 2,
                          ' Separated': 3,
                          ' Widowed': 4,
                          ' Married-spouse-absent': 5,
                          ' Married-AF-spouse': 6
                          }
df_train_set['maritalStatus'] = df_train_set['maritalStatus'].map(marital_status_mapping)
df_test_set['maritalStatus'] = df_test_set['maritalStatus'].map(marital_status_mapping)
# 对训练集、测试集同时处理离散型属性occupation
occupation_mapping = {' Tech-support': 0,
                      ' Craft-repair': 1,
                      ' Other-service': 2,
                      ' Sales': 3,
                      ' Exec-managerial': 4,
                      ' Prof-specialty': 5,
                      ' Handlers-cleaners': 6,
                      ' Machine-op-inspct': 7,
                      ' Adm-clerical': 8,
                      ' Farming-fishing': 9,
                      ' Transport-moving': 10,
                      ' Priv-house-serv': 11,
                      ' Protective-serv': 12,
                      ' Armed-Forces': 13
                      }
df_train_set['occupation'] = df_train_set['occupation'].map(occupation_mapping)
df_test_set['occupation'] = df_test_set['occupation'].map(occupation_mapping)
# 对训练集、测试集同时处理离散型属性relationship
relationship_mapping = {' Wife': 0,
                        ' Own-child': 1,
                        ' Husband': 2,
                        ' Not-in-family': 3,
                        ' Other-relative': 4,
                        ' Unmarried': 5
                        }
df_train_set['relationship'] = df_train_set['relationship'].map(relationship_mapping)
df_test_set['relationship'] = df_test_set['relationship'].map(relationship_mapping)
# 对训练集、测试集同时处理离散型属性race
race_mapping = {' White': 0,
                ' Asian-Pac-Islander': 1,
                ' Amer-Indian-Eskimo': 2,
                ' Other': 3,
                ' Black': 4
                }
df_train_set['race'] = df_train_set['race'].map(race_mapping)
df_test_set['race'] = df_test_set['race'].map(race_mapping)
# 对训练集、测试集同时处理离散型属性sex
sex_mapping = {' Female': 0,
               ' Male': 1,
               }
df_train_set['sex'] = df_train_set['sex'].map(sex_mapping)
df_test_set['sex'] = df_test_set['sex'].map(sex_mapping)
# 对训练集、测试集同时处理离散型属性native-country
native_country_mapping = {' United-States': 0,
                          ' Cambodia': 1,
                          ' England': 2,
                          ' Puerto-Rico': 3,
                          ' Canada': 4,
                          ' Germany': 5,
                          ' Outlying-US(Guam-USVI-etc)': 6,
                          ' India': 7,
                          ' Japan': 8,
                          ' Greece': 9,
                          ' South': 10,
                          ' China': 11,
                          ' Cuba': 12,
                          ' Iran': 13,
                          ' Honduras': 14,
                          ' Philippines': 15,
                          ' Italy': 16,
                          ' Poland': 17,
                          ' Jamaica': 18,
                          ' Vietnam': 19,
                          ' Mexico': 20,
                          ' Portugal': 21,
                          ' Ireland': 22,
                          ' France': 23,
                          ' Dominican-Republic': 24,
                          ' Laos': 25,
                          ' Ecuador': 26,
                          ' Taiwan': 27,
                          ' Haiti': 28,
                          ' Columbia': 29,
                          ' Hungary': 30,
                          ' Guatemala': 31,
                          ' Nicaragua': 32,
                          ' Scotland': 33,
                          ' Thailand': 34,
                          ' Yugoslavia': 35,
                          ' El-Salvador': 36,
                          ' Trinadad&Tobago': 37,
                          ' Peru': 38,
                          ' Hong': 39,
                          ' Holand-Netherlands': 40
                          }
df_train_set['nativeCountry'] = df_train_set['nativeCountry'].map(native_country_mapping)
df_test_set['nativeCountry'] = df_test_set['nativeCountry'].map(native_country_mapping)
# 对训练集、测试集同时处理离散型属性income
income_mapping = {' <=50K': 0,
                  ' >50K': 1,
                  ' <=50K.': 0,
                  ' >50K.': 1,
                  }
df_train_set['income'] = df_train_set['income'].map(income_mapping)
df_test_set['income'] = df_test_set['income'].map(income_mapping)

# 将预处理后的训练集与测试集数据输出到csv文件
df_train_set.to_csv('./train_adult_processed.csv', index=False)
df_test_set.to_csv('./test_adult_processed.csv', index=False)

columns = list(df_train_set.columns)


def calc_gini(df):
    """
    计算数据集的基尼指数
    :param df: 数据集
    :return: 基尼指数
    """
    p0 = 0
    n = 0
    for num in df['income']:
        if num == 0:
            p0 += 1
        n += 1
    p0 = p0 / n
    p1 = 1 - p0
    return 1 - p0 * p0 - p1 * p1


def split_dataset(df, index, value):
    """
    按照给定的列划分数据集
    :param df: 原始数据集
    :param index: 指定特征的列索引
    :param value: 指定特征的值
    :return: 切分后的数据集(left_df, right_df)
    """
    # 将数据集划分为两半，分发给左子树和右子树
    # index对应离散型特征时，左子树为符合value的子集，右子树为不符合value的子集
    # index对应连续型特征时，左子树为小于等于value的子集，右子树为大于value的子集
    feature = columns[index]
    if feature in discrete_column:
        left_df = df[df[feature] == value]
        right_df = df[df[feature] != value]
    else:
        left_df = df[df[feature] <= value]
        right_df = df[df[feature] > value]
    return left_df, right_df


def choose_best_feature_to_split(df):
    """
    选择最好的特征进行分裂
    :param df: 数据集
    :return: best_value:(分裂特征的index, 特征的值), best_df:(分裂后的左右子树数据集), min_gini:(选择该属性分裂的最小基尼指数)
    """
    best_value = ()
    min_gini = calc_gini(df)
    best_df = ()
    for index in range(len(columns) - 1):  # 最后一列是income，因此要减1
        feature = columns[index]
        for val in set(df[feature].values):
            left_df, right_df = split_dataset(df, index, val)
            left_size = len(left_df)
            right_size = len(right_df)
            if left_size == 0 or right_size == 0:
                continue
            total_size = left_size + right_size
            left_gini = calc_gini(left_df)
            right_gini = calc_gini(right_df)
            new_gini = left_gini * left_size / total_size + right_gini * right_size / total_size
            if new_gini < min_gini:
                min_gini = new_gini
                best_value = index, val
                best_df = left_df, right_df
    return best_value, best_df, min_gini


def build_decision_tree(df, layer, max_layer):
    """
    构建CART树
    :param df: 数据集
    :param layer: 当前所在层数(从0开始计算)
    :param max_layer: 剪枝时所允许的最大层数(为-1时则不减枝)
    :return: CART树
    """
    best_value, best_df, min_gini = choose_best_feature_to_split(df)
    # CART树表示为[leaf_flag, label, left_tree, right_tree, best_value, layer]
    # 其中leaf_flag标记是否为叶子
    if len(set(df['income'])) == 1:  # 若income的取值只有一种，说明已分“纯”
        cart = np.array([1, list(df['income'])[0], None, None, (), layer], dtype=object)
        return cart  # 递归结束情况1: 若当前集合的所有样本标签相等,即样本已被分"纯",则可以返回该标签值作为一个叶子节点
    elif best_value == () or (max_layer != -1 and layer >= max_layer):
        # 若best_value为(), 说明已经没有可用的特征; 若layer >= max_layer且max_layer不为-1, 说明需要剪枝
        if sum(df['income']) > (len(df['income']) - sum(df['income'])):
            label = 1
        else:
            label = 0
        cart = np.array([1, label, None, None, (), layer], dtype=object)
        return cart  # 递归结束情况2: 若当前训练集的所有特征都被使用完毕或需要剪枝, 则返回样本最多的标签作为结果
    else:
        left_tree = build_decision_tree(best_df[0], layer + 1, max_layer)
        right_tree = build_decision_tree(best_df[1], layer + 1, max_layer)
        cart = np.array([0, -1, left_tree, right_tree, best_value, layer], dtype=object)
        return cart


def save_decision_tree(cart):
    """
    决策树的存储
    :param cart: 训练好的决策树
    :return: void
    """
    np.save('cart.npy', cart)


def load_decision_tree():
    """
    决策树的加载
    :return: 保存的决策树
    """

    cart = np.load('cart.npy', allow_pickle=True)
    return cart


def classify(cart, df_row):
    """
    用训练好的决策树进行分类
    :param cart:决策树模型
    :param df_row: 一条测试样本
    :return: 预测结果
    """
    while cart[0] != 1:
        index, value = cart[4]
        feature = columns[index]
        if feature in discrete_column:
            if df_row[feature] == value:
                cart = cart[2]
            else:
                cart = cart[3]
        else:
            if df_row[feature] <= value:
                cart = cart[2]
            else:
                cart = cart[3]
    return cart[1]


def predict(cart, df):
    """
    用训练好的决策树进行分类
    :param cart:决策树模型
    :param df: 所有测试集
    :return: 预测结果
    """
    pred_list = []
    for i in range(len(df)):
        pred_label = classify(cart, df.iloc[i, :])
        if pred_label == -1:
            pred_label = random.randint(0, 1)  # 防止classify执行到返回-1,但一般不会执行到返回-1
        pred_list.append(pred_label)
    return pred_list


def calc_acc(pred_list, test_list):
    """
    返回预测准确率
    :param pred_list: 预测列表
    :param test_list: 测试列表
    :return: 准确率
    """
    pred = np.array(pred_list)
    test = np.array(test_list)
    acc = np.sum(pred_list == test_list) / len(test_list)
    return acc


for max_layer in range(-1, 100):
    if max_layer == -1:
        print("不对决策树的高度进行限制：")
    else:
        print("限制决策树的最大层数为%d(从0开始编号)：" % max_layer)
    cart = build_decision_tree(df_train_set, 0, max_layer)
    pred_list1 = predict(cart, df_train_set)
    pred_list2 = predict(cart, df_test_set)
    acc1 = calc_acc(pred_list1, df_train_set['income'].to_numpy())
    acc2 = calc_acc(pred_list2, df_test_set['income'].to_numpy())
    print("此时在训练集上的准确率为{}，在测试集上的准确率为{}\n".format(acc1, acc2))
