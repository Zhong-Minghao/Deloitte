import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import LocalOutlierFactor

# ------------------------版本及路径设置-----------------------
version = "1.f"
date = "20220822"
original_data_1 = 'df_factor_01_20220822'
original_data_2 = 'df_factor_02_20220822'
old_range_data = r'./input/[0] 老阈值.xlsx'

# ------------------------- 参数设置 -------------------------

# 判断数据零的比例，大于此比例则进行去零操作
zero_percentage = 0.15

# 调整零的个数至所有数字的比例(这里设置为和触发值(zero_percentage)一样)
targeted_zero_percentage = 0.15

# 正态检验置信度alpha
alpha = 0.05

# 按顺序分别为: z-score参数, LOF的分类阈值
# 目标异常比例分别为15%, 5%
parameter_dict = {1: [1.44, 1.5], 2: [2.82, 3]}

#
asserted_zero = targeted_zero_percentage / (1 - targeted_zero_percentage)
z_score_parameter_1, my_method_1 = parameter_dict[1]
z_score_parameter_2, my_method_2 = parameter_dict[2]

# k值计算的分母,挪到下面
my_k_denominator = 3

# LOF初筛使用阈值
initial_filter_m = 500

# ------------------------- 阈值计算函数 -------------------------


def read_df(df_name):
    return eval(f"pd.read_csv(r'./dfs/{df_name}.txt', delimiter = ',')")


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


def type_test(series):
    """
    检验序列的分布情况。
    首先对原序列进行KS检验,判断是否为正态分布。若不通过，进行BOX-COX变换，再次检验是否符合正态分布。

    :param series: 待检验的序列
    :return: 代表判断结果的数字
    """
    series = series.dropna()
    p = stats.kstest(series, 'norm')
    if p[1] > alpha:
        return 1
    else:
        # +1是为了防止浮点数溢出导致的series中存在负数
        series = series - series.min() + 1
        # Box-cox不能处理均匀分布
        if len(series.unique()) != 1:
            # lamda=None,自动匹配；alpha为置信度
            y_0, lambda_0 = stats.boxcox(series, lmbda=None, alpha=None)
            p = stats.kstest(y_0, 'norm')
            if p[1] > alpha:
                return 2
            else:
                return 3
        else:
            return 4


def z_score(series, z_score_parameter):
    """
    对服从正态分布的序列进行分析,阈值为均值加减n倍的标准差，并计算值域

    :param z_score_parameter: z-score
    :param series: 待检验的序列
    :return: 异常值判定结果序列 outlier
    """
    this_z_score = (series - np.mean(series)) / np.std(series)
    outlier = (this_z_score < np.mean(series) - z_score_parameter * np.std(series)) | (
            this_z_score > np.mean(series) + z_score_parameter * np.std(series))
    return outlier


def local_outlier_factor(data, predict, k):
    """
    计算LOF值

    :param data: 训练样本
    :param predict: 测试样本
    :param k: k值
    :return: 每一个测试样本 LOF 值及对应的第 k 距离
    """
    clf = LocalOutlierFactor(n_neighbors=k + 1, contamination=0.1, n_jobs=-1, novelty=True)
    clf.fit(data)
    predict_to_use = predict.copy()
    predict_to_use['local outlier factor'] = - clf.score_samples(predict)
    return predict_to_use


def get_predict_result(my_data: pd.Series, k_denominator: int):
    """ 获取LOF离群因子计算结果 """
    data_two_dim = list(zip(list(my_data.copy().to_frame().iloc[:, 0]), np.zeros_like(my_data)))
    predict_to_use = my_data.to_frame()
    predict_to_use['0'] = 0
    return local_outlier_factor(data_two_dim, predict_to_use, int(len(my_data) / k_denominator)).drop('0', axis=1)


def do_lof_judge(predict: pd.DataFrame, this_method: float):
    """ 根据method给出异常值判断 """
    predict_to_use = predict.copy()
    predict_to_use.loc[predict_to_use['local outlier factor'] > this_method, "abnormal"] = True
    predict_to_use.loc[predict_to_use['local outlier factor'] <= this_method, "abnormal"] = False
    return predict_to_use


def drop_zero(series):
    """ 特殊处理：当0的值过多的时候，调整0的数量至目标比例 """
    # 当序列全为0时，不需要去零，直接全部LOF，无异常值
    if (series == 0).sum() != series.count():
        this_name = series.name
        series_non_zero = series.loc[(series != 0)]
        series_zero = series.loc[(series == 0)]
        zeros_num = int(asserted_zero * len(series))
        series_zero = series_zero.iloc[0:zeros_num]
        series = pd.concat([series_non_zero, series_zero])
        series.name = this_name
    return series


if __name__ == '__main__':
    # ------------------------- 数据读取和初始化 -------------------------

    # 读取数据
    df_1 = read_df(original_data_1)
    df_2 = read_df(original_data_2)
    df = pd.merge(df_1, df_2, on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'], how='left')
    del df_1
    del df_2

    # 读取老阈值表格信息
    origin_threshold = pd.read_excel(old_range_data, header=1, sheet_name='转格式',
                                     usecols=['细分行业', '指标代码', '方向\n（单向/双向）']).rename(columns={'细分行业': 'industry',
                                                                                              '指标代码': 'factor',
                                                                                              '方向\n（单向/双向）': 'direction'})

    # 提取需计算的指标
    factors_contain = list(origin_threshold['factor'].unique())
    factors_to_cal = [x for x in factors_contain if not (x.endswith('_01_01') or x.endswith('_01_02'))]
    basic_cols = [x for x in df.columns if 'BASIC_' in x or 'PROFILE_' in x]
    basic_cols.extend(factors_to_cal)
    df = df[basic_cols]
    # 去除原数据中的inf
    df = df.replace([np.inf, -np.inf], np.nan)
    print("已完成数据读取")

    # 生成行业清单
    industries_list = [x for x in list(df['PROFILE_industry'].unique()) if x != '']
    # industries_list = ['传媒']

    # 初始化输出大表
    output_df = origin_threshold[['industry', 'factor', 'direction']]
    output_df['num_nan'] = np.nan
    output_df['num_zero'] = np.nan
    output_df['num_superfluous_zero'] = np.nan
    output_df['num_extreme'] = np.nan
    output_df['num_effective'] = np.nan
    output_df['trigger_zero'] = np.nan
    output_df['method'] = ''
    output_df['parameters'] = ''
    output_df['lower_1'] = np.nan
    output_df['upper_1'] = np.nan
    output_df['num_ab_1'] = np.nan
    output_df['perc_ab_1'] = np.nan
    output_df['lower_2'] = np.nan
    output_df['upper_2'] = np.nan
    output_df['num_ab_2'] = np.nan
    output_df['perc_ab_2'] = np.nan

# ---------------------------------------------------------------

    for this_industry in industries_list:
        print('=' * 50)
        print(f'正在处理：{this_industry}')
        for this_factor in factors_contain:
            print(f'    {this_factor}')

            # 计算中使用的指标名
            factor_name = this_factor.replace('_01_', '_').replace('_02_', '_')

            # 读取所需数据
            this_sample = df.loc[df['PROFILE_industry'] == this_industry, factor_name]

            #
            num_initial = len(this_sample)
            num_nan = (this_sample.isnull()).sum()
            num_zero = (this_sample == 0).sum()

            # version 1: 剔空
            tmp_series_1 = this_sample.dropna()
            if len(tmp_series_1) < 2:
                continue

            # version 2: 控0
            if num_zero / (num_initial - num_nan) > zero_percentage:
                trigger_zero = True
                tmp_series_2 = drop_zero(tmp_series_1)
            else:
                trigger_zero = np.nan
                tmp_series_2 = tmp_series_1.copy()

            #
            num_superfluous_zero = len(tmp_series_1) - len(tmp_series_2)

            # version 3: LOF初筛
            lof_values = get_predict_result(tmp_series_2, my_k_denominator)
            lof_result_pre = do_lof_judge(lof_values, initial_filter_m)
            tmp_series_3 = lof_result_pre.loc[lof_result_pre['abnormal'] == False, factor_name]

            #
            num_extreme = lof_result_pre['abnormal'].sum()
            num_effective = num_initial - num_nan - num_superfluous_zero - num_extreme

            # 判断序列类型
            type_result = type_test(tmp_series_3)
            """
                序列类型说明：
                    1：经BOX-COX转化前，正态分布
                    2：经BOX-COX转化后，正态分布
                    3：经BOX-COX转化后，非正态分布
                    4：均匀分布
            """

            # 对分类为’1‘的序列的处理
            if type_result == 1:
                method_to_use = '正态分布'
                anomaly_result_1 = z_score(tmp_series_3, z_score_parameter_1)
                anomaly_result_2 = z_score(tmp_series_3, z_score_parameter_2)

            # 对分类为’2‘的序列的处理
            elif type_result == 2:
                method_to_use = '正态分布'
                tmp_series_3 = tmp_series_3 - tmp_series_3.min() + 1
                y, lambda0 = stats.boxcox(tmp_series_3, lmbda=None, alpha=None)
                tmp_series_4 = pd.Series(y)
                tmp_series_4.index = tmp_series_3.index
                anomaly_result_1 = z_score(tmp_series_4, z_score_parameter_1)
                anomaly_result_2 = z_score(tmp_series_4, z_score_parameter_2)

            # 对分类为’3‘的序列的处理
            elif type_result == 3:
                method_to_use = 'LOF'
                lof_result = get_predict_result(tmp_series_3, my_k_denominator)
                anomaly_result_1 = do_lof_judge(lof_result, my_method_1)['abnormal']
                anomaly_result_2 = do_lof_judge(lof_result, my_method_2)['abnormal']

            # 对分类为’4‘的序列的处理
            elif type_result == 4:
                method_to_use = 'LOF'
                anomaly_result_1 = pd.Series(False, index=tmp_series_3.index)
                anomaly_result_2 = pd.Series(False, index=tmp_series_3.index)

            else:
                raise Exception('存在异常序列类型')

            this_direction = output_df.loc[(output_df['industry'] == this_industry)
                                           & (output_df['factor'] == this_factor), 'direction'].iloc[0]

            # 触发
            lower_1, upper_1 = get_range(tmp_series_3, anomaly_result_1)
            lower_2, upper_2 = get_range(tmp_series_3, anomaly_result_2)
            if this_direction == '向下':
                num_ab_1 = this_sample[this_sample < lower_1].count()
                num_ab_2 = this_sample[this_sample < lower_2].count()
            elif this_direction == '向上':
                num_ab_1 = this_sample[this_sample > upper_1].count()
                num_ab_2 = this_sample[this_sample > upper_2].count()
            else:
                num_ab_1 = this_sample[(this_sample < lower_1) | (this_sample > upper_1)].count()
                num_ab_2 = this_sample[(this_sample < lower_2) | (this_sample > upper_2)].count()
            perc_ab_1 = num_ab_1 / num_initial
            perc_ab_2 = num_ab_2 / num_initial

            # 记录统计
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_nan'] = num_nan
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_zero'] = num_zero
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_superfluous_zero'] = num_superfluous_zero
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_extreme'] = num_extreme
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_effective'] = num_effective
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'trigger_zero'] = trigger_zero
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'method'] = method_to_use
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'lower_1'] = lower_1
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'upper_1'] = upper_1
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_ab_1'] = num_ab_1
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'perc_ab_1'] = perc_ab_1
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'lower_2'] = lower_2
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'upper_2'] = upper_2
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'num_ab_2'] = num_ab_2
            output_df.loc[(output_df['industry'] == this_industry)
                          & (output_df['factor'] == this_factor), 'perc_ab_2'] = perc_ab_2

    output_df.loc[output_df['method'] == 'LOF', 'parameters'] = f'm普通={my_method_1}；m高危={my_method_2}'
    output_df.loc[output_df['method'] == '正态分布',
                  'parameters'] = f'z-score普通={z_score_parameter_1}；z-score高危={z_score_parameter_2}'

    output_df.to_excel(r'./output/[1] 阈值计算_v' + f'{version}_{date}.xlsx', index=False)
