import json
import random

import numpy as np
import pandas as pd


def View_data(df):
    """查看数据基本信息"""
    print(df.info())
    print(df.describe())
    print(df.describe(include=['O']))
    print(df.count())
    print(df.nunique())
    print(df.columns)
    print(df.count() / len(df))

def Delete_column(df, columns_to_drop):
    """删除指定的列"""
    new_df = df.drop(columns=columns_to_drop)
    return new_df

def Abnormal_value(df):
    """第二次清洗，检测异常值"""
    for column in df.columns:
        if df[column].dtype != 'object':
            df[column] = df[column].astype(float)  # 将列转换为浮点数类型
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[column] = df[column].where(df[column] >= lower_bound, lower_bound)
            df[column] = df[column].where(df[column] <= upper_bound, upper_bound)
    return df

def Fill_1(df):
    """填补空值,数值数据进行线性插值，文字数据使用众数填充"""
    for column in df.columns:
        if column != 'RRR':
            if df[column].dtype == 'object':
                # 对于非数值数据，使用众数填充缺失值，若全是缺失值，则填充为 0
                if df[column].isnull().all():
                    df[column] = df[column].fillna(0)
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            else:
                if df[column].isnull().all():
                    df[column] = df[column].fillna(0)
                else:
                    df[column] = df[column].interpolate()
                    df[column] = df[column].bfill().ffill()
    return df

def Fill_2(df):
    """填补空值,数值数据进行线性插值，文字数据使用众数填充"""
    for column in df.columns:
        if column != 'RRR':
            if df[column].dtype == 'object':
                # 对于非数值数据，使用众数填充缺失值，若全是缺失值，则填充为 0
                if df[column].isnull().all():
                    df[column] = df[column].fillna(0)
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            else:
                if df[column].isnull().all():
                    df[column] = df[column].fillna(0)
                else:
                    df[column] = df[column].interpolate(method='nearest')
                    df[column] = df[column].bfill().ffill()
    return df


def Encoding(df, mapping_file, threshold=100, label_column='RRR', skip_columns=['VV']):
    """创建映射字典，对数据类型是汉字且种类小于100的列的值进行数字编码"""
    mapping_dict = {}
    for column in df.columns:
        if column not in skip_columns and column != label_column and df[column].dtype == 'object' and df[column].nunique() < threshold:
            uniques = df[column].unique()
            mapping_dict[column] = {unique: float(i+1) for i, unique in enumerate(uniques)}  # 映射字典从 1 开始，转换为浮点数
    with open(mapping_file, 'w') as f:
        json.dump(mapping_dict, f, indent=4)
    return mapping_dict

def Mapping(df, mapping_file):
    """应用映射字典将分类数据转换为数值数据"""
    with open(mapping_file, 'r') as f:
        mapping_dict = json.load(f)
    mode_dict = df.mode().iloc[0]  # 计算每列的众数
    mode_mapping_dict = mode_dict.map(mapping_dict).fillna(0)  # 计算众数对应的映射值，如果不存在，填充为 0
    for column in df.columns:
        if column in mapping_dict:
            df[column] = df[column].map(mapping_dict[column]).fillna(mode_dict[column]).astype(float)  # 使用众数填充 NaN 值，转换为浮点数
        elif column == 'VV':
            df[column] = df[column].apply(lambda x: 0.0 if pd.isnull(x) or x == '低于 0.1' else float(x))
        elif column == 'RRR':
            df[column] = df[column].apply(lambda x: 0.0 if x == '无降水' else 1.0 if pd.notnull(x) else x)
    return df

def Compute_scaling(df,scaling_file, method='minmax', label_column='RRR'):
    """计算并保存特征缩放的参数"""
    scaling_dict = {}
    for column in df.columns:
        if df[column].dtype != 'object' and column != label_column:
            if method == 'minmax':
                min_val = df[column].min()
                max_val = df[column].max()
                scaling_dict[column] = {'min': min_val, 'max': max_val}
            elif method == 'standard':
                mean_val = df[column].mean()
                std_val = df[column].std()
                scaling_dict[column] = {'mean': mean_val, 'std': std_val}
            else:
                raise ValueError('Unknown method: ' + method)
    with open(scaling_file, 'w') as f:
        json.dump(scaling_dict, f, indent=4)
    return scaling_dict

def Apply_scaling(df, scaling_file):
    """读取并应用特征缩放的参数"""
    with open(scaling_file, 'r') as f:
        scaling_dict = json.load(f)
    for column in df.columns :
        if column in scaling_dict:
            if 'min' in scaling_dict[column] and 'max' in scaling_dict[column]:  # min-max scaling
                min_val = scaling_dict[column]['min']
                max_val = scaling_dict[column]['max']
                if max_val != min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
            elif 'mean' in scaling_dict[column] and 'std' in scaling_dict[column]:  # standardization
                mean_val = scaling_dict[column]['mean']
                std_val = scaling_dict[column]['std']
                if std_val != 0:
                    df[column] = (df[column] - mean_val) / std_val
    return df

def Split(df):
    """将数据分割为训练数据和测试数据"""
    # 计算测试数据的数量
    test_size = int(0.2 * len(df))
    # 随机选择一部分行作为测试数据
    test_indices = random.sample(range(len(df)), test_size)
    test_df = df.iloc[test_indices]
    # 将剩余的行作为训练数据
    train_df = df.drop(test_indices)
    return train_df, test_df

def df_to_json(df, output_file, label_column='RRR'):
    """将 DataFrame 转换为 JSON 文件，标签列在最后"""
    df = df.iloc[:, 1:]
    cols = df.columns.tolist()
    cols.remove(label_column)  # 移除标签列
    cols.append(label_column)  # 将标签列添加到末尾
    df = df[cols]  # 重新索引 DataFrame
    df.to_json(output_file, orient='records', lines=True)
    
def Kmeans(X, n_clusters, max_iters=1000):
    """聚类算法填补空缺标签"""
    # 确保 X 是数值型的
    X = X.astype(float)

    # 1. 随机选择初始聚类中心
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False)]

    for _ in range(max_iters):
        # 2. 计算每个样本到每个聚类中心的距离
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))

        # 3. 将每个样本分配到最近的聚类中心
        labels = np.argmin(distances, axis=0)

        # 4. 计算新的聚类中心
        new_centroids = np.array([X[labels==k].mean(axis=0) for k in range(n_clusters)])

        # 5. 如果聚类中心没有变化，那么算法已经收敛
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

def Cluster(df, n_clusters, label_column):
    # 分离有标签的部分和无标签的部分
    df_labeled = df[pd.notnull(df[label_column])]
    df_unlabeled = df[pd.isnull(df[label_column])]

    # 删除第一列
    df_labeled = df_labeled.iloc[:, 1:]
    df_unlabeled = df_unlabeled.iloc[:, 1:]

    # 使用有标签的部分训练一个K-means模型
    labels, centroids = Kmeans(df_labeled.drop(label_column, axis=1).values, n_clusters)

    # 计算每个聚类中的样本的平均标签
    avg_labels = [df_labeled[labels == i][label_column].mean() for i in range(n_clusters)]

    # 根据平均标签，给每个聚类中心分配一个标签
    centroid_labels = [1 if avg_label > 0.5 else 0 for avg_label in avg_labels]

    # 使用训练好的模型预测无标签部分的标签
    labels_unlabeled = [centroid_labels[i] for i in np.argmin(np.sqrt(((df_unlabeled.drop(label_column, axis=1).values - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)]

    # 将预测的标签填充到原始数据中
    df.loc[pd.isnull(df[label_column]), label_column] = labels_unlabeled

    return df


def main(input_file, output_train_file, output_test_file,  mapping_file, scaling_file, label_column='RRR',columns_to_drop = ['ff10', 'Tg', 'E\'', 'sss']):
    """_summary_
    1. 先读入excel文件
    2. 丢弃一些列
    3. 将异常值替换为上下界
    4. 对非标签列插值 (数值数据进行线性插值，文字数据使用众数填充)
    5. 对文字数据应用编码
    6. 用聚类算法填补空缺标签
    7. 缩放特征 (minmax缩放)
    8. 分割数据集
    """
    df = pd.read_excel(input_file)
    #View_data(df)
    
    df = Delete_column(df, columns_to_drop)
    
    df = Abnormal_value(df)

    df = Fill_1(df)
    #df = Fill_2(df)
   
    
    # Encoding(df, mapping_file)  # 创建并保存映射字典
    df = Mapping(df, mapping_file)  # 应用映射字典
    
    print("\nData after filling missing values:")
    print(df)
    
    Compute_scaling(df, scaling_file)  # 创建并保存特征缩放的参数
    df = Apply_scaling(df, scaling_file)  # 应用特征缩放的参数

    # 聚类算法填补空缺标签
    Cluster(df, n_clusters=2, label_column=label_column)
    
    # 分割数据集并保存为 JSON 文件
    train_df, test_df = Split(df)
    df_to_json(train_df, output_train_file)
    df_to_json(test_df, output_test_file)
    
    
    # prompt
    print('Done!')


# 定义文件路径

input_file = 'Big_HW/training_dataset.xls'
#input_file = 'Big_HW/test.xls'
output_train = 'Big_HW/train_1.json'
output_test = 'Big_HW/test_1.json'

scaling_file = 'Big_HW/scaling_dict.json'
mapping_file = 'Big_HW/mapping_dict.json'


# 调用主函数
main(input_file, output_train, output_test, mapping_file, scaling_file)
