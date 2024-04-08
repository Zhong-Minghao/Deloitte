import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import os
from matplotlib.ticker import FuncFormatter

# ------------------------- 参数设置 -------------------------

# 自动读取程序路径
root_path = r"./"

# 参数设置
# 运算过程中是否生成LOF结果分布图像；图像将保存在根目录的"figure"文件夹下
my_plot = False
# 是否进行正态分布数组与LOF的对比图，图像将保存在根目录的"comparison"文件夹下
my_comparison = False

# 严格等级的档位
strictness_level = 3
# 按顺序分别为: z-score参数, LOF的分类阈值
# 目标异常比例分别为15%,10%,5%
parameter_dict = {1: [1.44, 2.8],
                  2: [1.65, 6],
                  3: [2.82, 8]}
z_score_parameter, my_method = parameter_dict[strictness_level]

# 正态检验置信度alpha
alpha = 0.05
# 调整零的个数至所有数字的比例
targeted_zero_percentage = 0.15
asserted_zero = targeted_zero_percentage/(1-targeted_zero_percentage)
# 判断数据零的比例，大于此比例则进行去零操作
zero_percentage = 0.15
# k值计算的分母
my_k_denominator = 3
# 画图时源数据的百分位范围
axis_range = [5, 95]

# LOF初筛使用阈值
initial_filter_m = 500


# ------------------------- 数据读取和初始化 -------------------------

# 保存数据
def save_df(df_name, output_name):
    exec("{}.to_csv(root_path + '{}.csv', index=True, sep=',', float_format='%.16f')".format(df_name, output_name))


# 读取数据
df_1 = pd.read_csv(r'./dfs/df_factor_01_20220816.txt')
df_2 = pd.read_csv(r'./dfs/df_factor_02_20220816.txt')
df = pd.merge(df_1, df_2, left_on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'], right_on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'], how='left')

# 仅使用2017-2021年的数据
# df = df.loc[(df['BASIC_year'] >= 2017) & (df['BASIC_year'] <= 2021)].copy()

# 提取需计算字段
df_to_cal = pd.read_excel(root_path + '需计算阈值字段列表.xlsx', sheet_name="Sheet1")

# 提取需计算的指标
my_columns = df_to_cal['需计算字段'].tolist()
my_columns = [x for x in my_columns if not (x.endswith('_01_01') or x.endswith('_02_01'))]
basic_cols = [x for x in df.columns if 'BASIC_' in x or 'PROFILE_' in x]
basic_cols.extend(my_columns)
df = df[basic_cols]
print("已完成：数据读取")

# 去除原数据中的inf
df = df.replace([np.inf, -np.inf], np.nan)

# 预生成结果储存表格
result_df = df.copy()
result_df = result_df.iloc[:, 4:]

industries_list = [x for x in list(df['PROFILE_industry'].unique()) if x != '']

# 生成图片存储路径
if my_plot:
    os.makedirs(r'./figure', exist_ok=True)
    for this_industry in industries_list:
        os.makedirs(fr'./figure/{this_industry}', exist_ok=True)
if my_comparison:
    os.makedirs(r'./comparison', exist_ok=True)
    test_LOF_lower_range_df = pd.DataFrame(columns=industries_list, index=df.columns)
    test_LOF_upper_range_df = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成阈值存储表格
lower_range_df = pd.DataFrame(columns=industries_list, index=df.columns)
upper_range_df = pd.DataFrame(columns=industries_list, index=df.columns)

# 分布的类型表储存格
type_df = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成异常值占比储存表格
anomaly_proportion = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成零值占比储存表格
zeros_percentage = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成预处理占比储存表格
pre_process = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成样本数量储存表格
sample_num = pd.DataFrame(columns=industries_list, index=df.columns)

# 生成特殊处理的输出表格
iterables = [industries_list, lower_range_df.index.tolist()]
arrays = pd.MultiIndex.from_product(iterables, names=['industry', 'financial_index'])
zero_warning = pd.DataFrame(columns=['warning', 'refilled lower range', 'refilled upper range', 'refilled percentage'],
                            index=arrays)


# ------------------------- 计算相关函数 -------------------------

def get_range(data_series, outlier_series):
    """
    当前异常值结果被划分为两头异常，中间正常的状态。用此函数获取判定为异常的阈值
    ！注意这个函数直接修改了不同的全局变量！

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
    【1:经BOX-COX转化前,正态分布; 2:经BOX-COX转化后, 正态分布; 3:经BOX-COX转化后，非正态分布; 4:均匀分布】
    """
    series = series.dropna()
    p = stats.kstest(series, 'norm')
    if p[1] > alpha:
        return 1
    else:
        # +1是为了防止浮点数溢出导致的series中存在负数
        series = series - series.min() + 1
        # Boxcox不能处理均匀分布
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


def z_score(series):
    """
    对服从正态分布的序列进行分析,阈值为均值加减n倍的标准差，并计算值域

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
    clf = LocalOutlierFactor(n_neighbors=k + 1, algorithm='auto',
                             contamination=0.1, n_jobs=-1, novelty=True)
    clf.fit(data)
    # 获得LOF值
    predict['local outlier factor'] = - clf.score_samples(predict)
    return predict


def lof(data, predict: pd.DataFrame, k, method):
    """
    计算LOF值

    :param data: 训练样本
    :param predict: 测试样本
    :param k: k值，第几邻域
    :param method: 离群阈值，LOF大于该值认为是离群值
    :return: 离群点、正常点的分类情况
    """
    predict_to_use = predict.copy()
    predict_to_use["0"] = 0
    # 计算 LOF 离群因子
    predict_to_use = local_outlier_factor(data, predict_to_use, k)
    # 根据阈值划分离群点与正常点
    predict_to_use.loc[predict_to_use['local outlier factor'] > method, "离群点判断"] = True
    predict_to_use.loc[predict_to_use['local outlier factor'] <= method, "离群点判断"] = False
    return predict_to_use["离群点判断"]


def do_lof_test(my_data: pd.Series, this_method: float = my_method):
    """
    计算LOF值

    :param my_data: 原数据序列
    :param this_method: LOF的m值
    :return: 离群点、正常点的分类情况
    """
    data_two_dim = list(zip(list(my_data.copy().to_frame().iloc[:, 0]), np.zeros_like(my_data)))
    my_outlier = lof(data_two_dim, my_data.to_frame(), k=int(len(my_data) / my_k_denominator),
                     method=this_method)
    return my_outlier


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


def do_compare(series):
    norm_LOF_result = do_lof_test(series, my_method)
    # 上式画图会覆盖掉z-score图
    test_LOF_lower_range_df[this_industry][tmp_series_0.name], \
    test_LOF_upper_range_df[this_industry][tmp_series_0.name] = get_range(tmp_series_2, norm_LOF_result)
    graph_figures(tmp_series_3, True)


def to_percent(y, position):
    return str(round(100 * y, 2)) + "%"


def layout_conversion(lower_df, upper_df, o_percent_df):
    """
    输入存好的上下限DataFrame(以行业为列，指标为行)，调整布局

    :param lower_df: 下阈值表格
    :param upper_df: 上阈值表格
    :param o_percent_df: 异常值占比表格
    :return: 改变布局后的表格df
    """
    method_selected_df = type_df.replace([0, 1, 2, 3, 4, 5], ['LOF', 'z-score', 'z-score', 'LOF', 'LOF', ''])
    l_range = pd.DataFrame(lower_df.to_numpy().reshape(-1, 1, order='F'))
    u_range = pd.DataFrame(upper_df.to_numpy().reshape(-1, 1, order='F'))
    p_process = pd.DataFrame(pre_process.to_numpy().reshape(-1, 1, order='F'))
    o_percentage = pd.DataFrame(o_percent_df.to_numpy().reshape(-1, 1, order='F'))
    z_percentage = pd.DataFrame(zeros_percentage.to_numpy().reshape(-1, 1, order='F'))
    type_series = pd.DataFrame(method_selected_df.to_numpy().reshape(-1, 1, order='F'))
    samp_series = pd.DataFrame(sample_num.to_numpy().reshape(-1, 1, order='F'))
    merged_df = pd.concat([l_range, u_range, o_percentage, type_series, p_process, samp_series, z_percentage], axis=1)
    merged_df.index = arrays
    merged_df.columns = ['最终下限', '最终上限', '最终异常占比', 'z-score或LOF', '初筛中删除的样本数量', '样本数量', '0值占比']
    merged_df[zero_warning.columns.tolist()] = zero_warning
    return merged_df


# ------------------------- 画图相关函数 -------------------------

def graph_figures(my_series, comparison=False):
    """
    画出分布的直方图，给出阈值

    :param my_series: 某行业某指标的分布的数据
    :param comparison: 如果True,则输出正态分布序列z-score与LOF方法的对比图
    :return:
    """
    num_interval = 1000
    bound_1 = np.nanpercentile(my_series, axis_range[0])
    bound_2 = np.nanpercentile(my_series, axis_range[1])
    weights = [1. / len(my_series)] * len(my_series)
    my_bin = [x * (bound_2 - bound_1) / num_interval + bound_1 for x in range(0, num_interval + 1, 1)]

    plt.hist(my_series, bins=my_bin, weights=weights, color='yellowgreen', alpha=0.4)
    plt.title(f'distribution of {my_series.name}')
    fomatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    plt.grid(visible=True, axis='y', c='gainsboro')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pct_up = upper_range_df[this_industry][my_series.name]
    pct_low = lower_range_df[this_industry][my_series.name]
    plt.xlim(0, 1)
    # # 当去零后进行lof时，用新的阈值数据覆盖原阈值
    # if zero_refilled:
    #     zeros_array = pd.MultiIndex.from_product([[this_industry], [my_series.name]],
    #                                              names=['industry', 'financial_index'])
    #     pct_up = zero_warning.loc[zeros_array, ['refilled upper range']].values
    #     pct_low = zero_warning.loc[zeros_array, ['refilled lower range']].values

    l1 = ax.axvline(pct_up, color='olivedrab', ls='--')
    ax.axvline(pct_low, color='olivedrab', ls='--')

    # 当需要进行对比时，采用特殊处理
    if comparison:
        # 展示z-score和LOF阈值划分结果
        plt.title(f'{my_series.name} range given by z-score(green) and LOF(red)')
        pct_test_up = test_LOF_upper_range_df[this_industry][my_series.name]
        pct_test_low = test_LOF_lower_range_df[this_industry][my_series.name]
        l2 = ax.axvline(pct_test_up, ymax=0.9, color='firebrick', ls='--')
        ax.axvline(pct_test_low, ymax=0.9, color='firebrick', ls='--')
        plt.legend([l1, l2], ['range given by lof', 'range given by z-score'])
        plt.savefig(root_path + r"/comparison/" + my_series.name + "_" + ".png")
        # plt.show()
        plt.close()
    # 一般流程
    else:
        plt.legend([l1], ['range'])
        plt.savefig(root_path + f"/figure/{this_industry}/{my_series.name}.png")
        # plt.show()
        plt.close()
    return


# ------------------------- 初筛 -------------------------

# for this_industry in industries_list:
#     print(f'正在处理: {this_industry}……')
#     industry_df = df.loc[df["PROFILE_industry"] == this_industry].copy()
#     for this_indicator in my_columns:
#         print(this_indicator)
#         series_copy = industry_df[this_indicator].copy()
#         # 当原序列没有非空值或非空值小于2个时，不处理
#         if len(series_copy)-(series_copy.isnull().sum()) < 2:
#             continue
#         lof_result = do_lof_test(industry_df[this_indicator].copy(), initial_filter_m)
#         df.loc[lof_result[lof_result[:] == True].index.tolist(), this_indicator] = np.nan
#         pre_process[this_industry][this_indicator] = lof_result.sum()
# print('初筛结束')

# ------------------------- 运行代码 -------------------------

for this_industry in industries_list:
    print(f'正在处理: {this_industry}……')
    industry_df = df[df['PROFILE_industry'] == this_industry]
    industry_df = industry_df.iloc[:, 3:]
    for this_indicator in my_columns:
        print(this_indicator)
        tmp_series_0 = industry_df[this_indicator]
        # version 1: 剔空
        tmp_series_1 = tmp_series_0.dropna()
        zeros_percentage[this_industry][this_indicator] = (tmp_series_1 == 0).sum() / tmp_series_1.count()
        sample_num[this_industry][this_indicator] = tmp_series_1.count()
        if len(tmp_series_1) < 2:
            continue

        # version 2: 控0
        if (tmp_series_1 == 0).sum() / tmp_series_1.count() > zero_percentage:
            tmp_series_2 = drop_zero(tmp_series_1)
            zeros_array = pd.MultiIndex.from_product([[this_industry], [tmp_series_1.name]],
                                                    names=['industry', 'financial_index'])
            zero_warning.loc[zeros_array, ['warning']] = f'0值占比超{zero_percentage}'
        else:
            tmp_series_2 = tmp_series_1.copy()

        # version 3: 初筛
        lof_result_i = do_lof_test(tmp_series_2, initial_filter_m)
        tmp_series_3 = tmp_series_2.copy()
        tmp_series_3.loc[lof_result_i[lof_result_i[:] == True].index.tolist()] = np.nan
        tmp_series_3 = tmp_series_3.dropna()
        pre_process[this_industry][this_indicator] = lof_result_i.sum()

        # 判断序列分布种类
        type_result = type_test(tmp_series_3)
        type_df[this_industry][this_indicator] = type_result
        sample_num[this_industry][this_indicator] = tmp_series_0.count()

        # 对分类为’1‘的序列的处理
        if type_result == 1:
            anomaly_result = z_score(tmp_series_3)
            result_df.loc[tmp_series_3.index.tolist(), this_indicator] = anomaly_result

            # if my_comparison:
            #     do_compare(tmp_series_3)

        # 对分类为’2‘的序列的处理
        elif type_result == 2:
            tmp_series_3 = tmp_series_3 - tmp_series_3.min() + 1
            y, lambda0 = stats.boxcox(tmp_series_3, lmbda=None, alpha=None)
            tmp_series_4 = pd.Series(y)
            tmp_series_4.index = tmp_series_3.index
            anomaly_result = z_score(tmp_series_4)
            result_df.loc[tmp_series_3.index.tolist(), this_indicator] = anomaly_result

            # if my_comparison:
            #     do_compare(tmp_series_3)

        # 对分类为’3‘的序列的处理
        elif type_result == 3:
            anomaly_result = do_lof_test(tmp_series_3, my_method)
            result_df.loc[tmp_series_3.index.tolist(), this_indicator] = anomaly_result

        # 对分类为’4‘的序列的处理
        elif type_result == 4:
            anomaly_result = pd.Series(False, index=tmp_series_3.index)
            # 这里贪图方便，索性将所有格（包括原本为空的格）填充为False
            result_df.loc[tmp_series_3.index.tolist(), this_indicator] = False

        else:
            raise Warning('KS检验步骤出现条件以外的情况！')

        l_range, u_range = get_range(tmp_series_3, anomaly_result)
        lower_range_df[this_industry][tmp_series_0.name], upper_range_df[this_industry][tmp_series_0.name] = l_range, u_range
        anomaly_proportion[this_industry][tmp_series_1.name] = (tmp_series_1[tmp_series_1 > u_range].count() +
                                                                tmp_series_1[tmp_series_1 < l_range].count()) / tmp_series_1.count()
        if my_plot:
            graph_figures(tmp_series_1)


range_df = layout_conversion(lower_range_df, upper_range_df, anomaly_proportion)

# 重新插入原表格左边13列的信息列
info = df.iloc[:, 0:3]
result_df = pd.concat([info, result_df], axis=1)

type_df = type_df.T

print('已计算完毕，正在保存文件……')

save_df('type_df', '序列分布检验结果_5_test')
save_df('result_df', '异常点检验结果_5_test')
save_df('range_df', '阈值&比例_5_test')
# save_df('anomaly_proportion', '异常比例_初筛')
# save_df('lower_range_df', '下限')
# save_df('upper_range_df', '上限')