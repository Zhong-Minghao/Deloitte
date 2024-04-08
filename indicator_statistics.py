import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

# ------------------------- 参数设置 -------------------------

# 自动读取程序路径
root_path = r"./"

# k值计算的分母
my_k_denominator = 3

# 调整零的个数至所有数字的比例
targeted_zero_percentage = 0.15
asserted_zero = targeted_zero_percentage/(1-targeted_zero_percentage)

# LOF初筛使用阈值
initial_filter_m = 500

# 百分位点
percentile_list = [5, 10, 15, 20, 25, 75, 80, 85, 90, 95]

# 标准差
standard_deviation_list = [-2, -1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 2]

# m值（从小到大排序）
m = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]


# ------------------------- 数据读取和初始化 -------------------------

# 保存数据
def save_df(df_name, output_name):
    exec("{}.to_csv(root_path + '{}.csv', index=True, sep=',', float_format='%.16f')".format(df_name, output_name))


# 读取数据
df_1 = pd.read_csv(root_path + r'/df_factor_01_20220822.txt')
df_2 = pd.read_csv(root_path + r'/df_factor_02_20220822.txt')
df = pd.merge(df_1, df_2, left_on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'],
              right_on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'], how='left')

# 读取字段
df_indicator = pd.read_excel(r'./old_range.xlsx')

df_indicator = df_indicator.set_index(['细分行业', '指标代码'])

industries_list = [x for x in list(df['PROFILE_industry'].unique()) if x != '']

output_df = pd.read_excel(r'./header.xlsx')


def drop_zero(series):
    """
    特殊处理：当0的值过多的时候，调整0的数量至目标比例

    :param series: 待处理序列
    :return: 处理后序列
    """
    # 当序列全为0时，不需要去零，直接全部LOF，无异常值
    if (series == 0).sum() != series.count():
        this_name = series.name
        series_non_zero = series.loc[(series != 0)]
        series_zero = series.loc[(series == 0)]
        zeros_num = int(asserted_zero * len(series))
        series_zero = series_zero.iloc[0:zeros_num]
        # 当asserted_zero为0时，这里会报警告，不过不影响,加一个0是防止序列只有1个非零值时，导致的序列只有一组坐标时不能使用LOF的情况
        series = series_non_zero.append(series_zero)
        series.name = this_name
    return series


def lof(data, predict: pd.DataFrame, k):
    """
    计算LOF值

    :param data: 训练样本
    :param predict: 测试样本
    :param k: k值，第几邻域
    :return: 离群点、正常点的分类情况
    """
    predict_to_use = predict.copy()
    predict_to_use["0"] = 0
    # 计算 LOF 离群因子
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto', contamination=0.1, n_jobs=-1, novelty=True)
    clf.fit(data)
    # 获得LOF值
    predict_to_use['local outlier factor'] = - clf.score_samples(predict_to_use)
    return predict_to_use


def do_lof_test(my_data: pd.Series):
    """
    计算LOF值

    :param my_data: 原数据序列
    :return: 离群点、正常点的分类情况
    """
    data_two_dim = list(zip(list(my_data.copy().to_frame().iloc[:, 0]), np.zeros_like(my_data)))
    my_lof = lof(data_two_dim, my_data.to_frame(), k=int(len(my_data) / my_k_denominator))
    return my_lof


def get_outlier(my_data, this_method):
    # 根据阈值划分离群点与正常点
    my_data.loc[my_data['local outlier factor'] > this_method, "离群点判断"] = True
    my_data.loc[my_data['local outlier factor'] <= this_method, "离群点判断"] = False
    return my_data['离群点判断']


def get_range(data_series, outlier_series):
    """
    当前异常值结果被划分为两头异常，中间正常的状态。用此函数获取判定为异常的阈值

    :param data_series:原序列
    :param outlier_series:对应的异常结果判定序列
    :return: 阈值的下限和上限
    """
    lower_limit = data_series[outlier_series == False].min(skipna=True)
    upper_limit = data_series[outlier_series == False].max(skipna=True)
    return lower_limit, upper_limit


i = 0
for this_industry in industries_list:
    print(f'正在处理: {this_industry}……')
    industry_df = df[df['PROFILE_industry'] == this_industry]
    industry_df = industry_df.iloc[:, 3:]
    for this_indicator in df_indicator.loc[this_industry].index:
        if "_01_" in this_indicator:
            this_indicator = this_indicator.replace("_01_", "_")
        print(this_indicator)
        result_list = [this_industry, this_indicator]
        tmp_series_0 = industry_df[this_indicator]

        # 初步处理
        tmp_series_1 = tmp_series_0.dropna()

        # 控0
        if (tmp_series_1 == 0).sum() / tmp_series_1.count() > targeted_zero_percentage:
            tmp_series_2 = drop_zero(tmp_series_1)
        else:
            tmp_series_2 = tmp_series_1.copy()

        this_lof = do_lof_test(tmp_series_2)
        lof_result = get_outlier(this_lof, initial_filter_m)
        tmp_series_3 = tmp_series_2.copy()
        tmp_series_3.loc[lof_result[lof_result == True].index.tolist()] = np.nan
        tmp_series_3 = tmp_series_3.dropna().copy()

        # 初步计数
        count_all = len(tmp_series_0)
        count_null = tmp_series_0.isnull().sum()
        series_mean = tmp_series_2.mean()
        series_median = tmp_series_2.median()
        series_var = tmp_series_2.var()
        count_zero = (tmp_series_1 == 0).sum()
        count_dropped_zero = len(tmp_series_1) - len(tmp_series_2)
        count_anomaly = lof_result.sum()
        count_available_sample = tmp_series_3.count()
        result_list.extend([count_null, count_all, np.nan, series_mean, series_median, series_var, count_zero,
                            count_dropped_zero, count_anomaly, count_available_sample, np.nan])
        # 如需取消输出，请运行下面这行
        # result_list.extend([np.nan] * 11)

        # 分位点
        for this_percentile in percentile_list:
            my_percentile = np.nanpercentile(tmp_series_3, this_percentile)
            result_list.append(my_percentile)
        # 如需取消输出，请运行下面这行
        # result_list.extend([np.nan] * len(percentile_list))

        # 标准差
        series_deviation = tmp_series_3.std()
        for this_deviation in standard_deviation_list:
            my_deviation = tmp_series_3.mean() + this_deviation * series_deviation
            result_list.append(my_deviation)
        # 如需取消输出，请运行下面这行
        # result_list.extend([np.nan] * 10)

        # m值边界点
        range_list = []
        this_lof = do_lof_test(tmp_series_3)
        for this_m in m:
            this_outlier_result = get_outlier(this_lof, this_m)
            l_range, u_range = get_range(tmp_series_3, this_outlier_result)
            range_list.insert(0, l_range)
            range_list.append(u_range)
        result_list.extend(range_list)
        # 如需取消输出，请运行下面这行
        # result_list.extend([np.nan] * 24)

        output_df.loc[i] = result_list
        i += 1

save_df('output_df', '统计结果')
