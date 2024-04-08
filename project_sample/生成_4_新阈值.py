import sys
from tool_widget import home_page_structure
from tool_widget import manual_selection_page_1
from tool_widget import table_widget_1
import pandas as pd
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QStackedLayout, QTableWidgetItem
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QBrush, QColor
import time
import matplotlib
matplotlib.use('agg')     # 设置plt后端参数，使其不显示画面,这句必须在这里，请勿调整
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import 生成_1_阈值计算 as Py1

#
version = Py1.version
date = Py1.date
original_data_1 = Py1.original_data_1
original_data_2 = Py1.original_data_2

# 在这行设置柱子的宽度
num_interval = 100

# ================================数据获取与参数设置部分=================================================

# 自动读取程序路径
root_path = r"./"
os.makedirs(r'./tmp', exist_ok=True)

this_path_1 = root_path + '/tmp/pic1.png'
this_path_2 = root_path + '/tmp/pic2.png'
this_path_3 = root_path + '/tmp/pic3.png'
this_path_4 = root_path + '/tmp/pic4.png'
this_path_5 = root_path + '/tmp/pic5.png'

# 读取原始数据
df1 = Py1.read_df(original_data_1)
df2 = Py1.read_df(original_data_2)

#
df = pd.read_excel(r"./output/[2] 阈值统计_v" + f"{version}_{date}.xlsx", header=0)

#
df['lower'] = np.nan
df['upper'] = np.nan

#
df.loc[df['type'] == "专家", 'lower'] = df.loc[df['type'] == "专家", 'old-lower']
df.loc[df['type'] == "专家", 'upper'] = df.loc[df['type'] == "专家", 'old-upper']

df.loc[df['new-type'] == "统计", 'lower'] = df.loc[df['new-type'] == "统计", 'new-lower']
df.loc[df['new-type'] == "统计", 'upper'] = df.loc[df['new-type'] == "统计", 'new-upper']

#
dff = df[(df['lower'].isnull()) | (df['upper'].isnull())].copy()

#
df_stats = pd.read_excel(r"./output/[3] 数据统计_v" + f"{version}_{date}.xlsx", header=0).rename(columns={
    "行业": "industry",
    "指标": "factor"
})
df_stats.columns = [str(x).replace(' ', '') for x in df_stats.columns]

#
df_combine = pd.merge(df1, df2, on=['BASIC_entity_name', 'BASIC_year', 'PROFILE_industry'], how='left')
df_combine = df_combine.replace([np.inf, -np.inf], np.nan)
print("已完成数据读取")

show_direction = {'向上': '→', '向下': '←', '双向': '← →'}


# =======================================函数================================================================
# ==========================================================================================================


def to_percent(y, position):
    return str(round(100 * y, 2)) + "%"


def alter_factor(this_factor):
    """
    指标特殊处理，将_01_改为_
    因为新数据中没有_01_的序列，因此df_stats，和df_combine中的取factor，用去除后的factor，否则用原来的
    """
    this_factor_origin = this_factor
    this_factor = this_factor.replace('_02_', '_').replace('_01_', '_')
    return this_factor_origin, this_factor


def get_anomaly_rate(series, series_direction, low_num, up_num):
    """计算阈值"""
    num_initial = len(series)
    if series_direction == '向下':
        if low_num == '0.01且>0':
            num_ab_1 = series[(series < 0.01) | (series > 0)].count()
        elif low_num == '-':
            num_ab_1 = '-'
        else:
            num_ab_1 = series[series < low_num].count()
    elif series_direction == '向上':
        num_ab_1 = series[series > up_num].count()
    else:
        if low_num == '0.01且>0':
            num_ab_1 = series[series > up_num].count() + series[(series < 0.01) | (series > 0)].count()
        elif low_num == '0.005且>0':
            num_ab_1 = series[series > up_num].count() + series[(series < 0.005) | (series > 0)].count()
        else:
            num_ab_1 = series[(series < low_num) | (series > up_num)].count()

    if num_ab_1 != '-':
        perc_ab_1 = num_ab_1 / num_initial
    else:
        perc_ab_1 = '-'

    return perc_ab_1


def graph_figures(my_industry, my_factor, indicator, pct_low, pct_up, pct_new_low=np.nan, pct_new_up=np.nan):
    """ 画出分布的直方图，给出阈值 """
    if pct_low == '0.01且>0':
        pct_low = float(pct_low.replace('0.01且>0', '0.01'))
    elif pct_low == '0.005且>0':
        pct_low = float(pct_low.replace('0.005且>0', '0.005'))
    plt.close()
    origin_my_factor, my_factor = alter_factor(my_factor)
    my_series = df_combine[df_combine['PROFILE_industry'] == my_industry][my_factor]
    direction = dff.loc[(dff['industry'] == my_industry) & (dff['factor'] == origin_my_factor), 'direction'].iloc[0]
    bound_1 = np.nanpercentile(my_series, 2)
    bound_2 = np.nanpercentile(my_series, 98)
    weights = [1. / len(my_series)] * len(my_series)
    my_bin = [x * (bound_2 - bound_1) / num_interval + bound_1 for x in range(0, num_interval + 1, 1)]
    plt.hist(my_series, bins=my_bin, weights=weights, color='yellowgreen', alpha=0.4)
    plt.title(f'range of {my_series.name} (direction:{show_direction[direction]})')
    fomatter = FuncFormatter(to_percent)
    plt.gca().yaxis.set_major_formatter(fomatter)
    plt.grid(visible=True, axis='y', c='gainsboro')
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if direction == '向下':
        l1 = ax.axvline(pct_low, color='firebrick', ls='--')
        l2 = ax.axvline(pct_new_low, color='olivedrab', ls='--')
    elif direction == '向上':
        l1 = ax.axvline(pct_up, color='firebrick', ls='--')
        l2 = ax.axvline(pct_new_up, color='olivedrab', ls='--')
    else:
        l1 = ax.axvline(pct_low, color='firebrick', ls='--')
        ax.axvline(pct_up, color='firebrick', ls='--')
        l2 = ax.axvline(pct_new_low, ymax=0.9, color='olivedrab', ls='--')
        ax.axvline(pct_new_up, ymax=0.9, color='olivedrab', ls='--')
    if indicator == 1:
        plt.legend([l1, l2], ['old range', 'new range'])
        plt.savefig(this_path_1)
    elif indicator == 2:
        plt.title(f'artificial range setting of {origin_my_factor}')
        plt.savefig(this_path_2)
    elif indicator == 3:
        plt.title(f'newly documented range setting of {origin_my_factor}')
        plt.legend([l1, l2], ['old range', ' documented range'])
        plt.savefig(this_path_3)
    return


def get_new_range(this_industry, this_factor, this_level):
    """跳转到新指标时，重新获取选项与原始新老阈值图片"""

    ui.select.artificial_lower = np.nan
    ui.select.artificial_upper = np.nan
    ui.select.listView.clear()

    origin_this_factor, this_factor = alter_factor(this_factor)
    # 基本统计信息：均值、中位数、方差 <—— df_stats
    ui.select.mean_display.setText(f"  {df_stats.loc[(df['industry'] == this_industry) & (df_stats['factor'] == this_factor), '均值'].iloc[0]}")
    ui.select.median_display.setText(f"  {df_stats.loc[(df['industry'] == this_industry) & (df_stats['factor'] == this_factor), '中位数'].iloc[0]}")
    ui.select.var_display.setText(f"  {df_stats.loc[(df['industry'] == this_industry) & (df_stats['factor'] == this_factor), '方差'].iloc[0]}")
    ui.select.direction_display.setText(f"  {df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor) & (df['level'] == this_level), 'direction'].iloc[0]}")
    ui.select.level_display.setText(f"  {df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor) & (df['level'] == this_level), 'level'].iloc[0]}")
    ui.select.old_upper_display.setText(f'  {df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "old-upper"].iloc[0]}'.replace('1000000000000000000', '-'))
    ui.select.old_lower_display.setText(f'  {df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "old-lower"].iloc[0]}'.replace('1000000000000000000', '-'))
    ui.select.old_anomaly_display.setText('  {:.5%}'.format(df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "old-perc_triggered"].iloc[0]))
    ui.select.new_upper_display.setText(f'  {df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "new-upper"].iloc[0]}')
    ui.select.new_lower_display.setText(f'  {df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "new-lower"].iloc[0]}')
    ui.select.new_anomaly_display.setText('  {:.5%}'.format(df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "new-perc_triggered"].iloc[0]))
    ui.select.note_display.setText(f'  {df.loc[(df["industry"] == this_industry) & (df["factor"] == origin_this_factor) & (df["level"] == this_level), "mark"].iloc[0]}')
    ui.select.current_industry_display.setText(this_industry)
    ui.select.current_factor_display.setText(origin_this_factor)

    # 数据分布直方图（含新老阈值） <——df1/df2 + df
    tmp_df = dff.loc[(dff['industry'] == this_industry) & (dff['factor'] == origin_this_factor) & (dff['level'] == this_level)]
    this_direction = tmp_df['direction'].iloc[0]
    range_low = tmp_df['old-lower'].iloc[0]
    range_up = tmp_df['old-upper'].iloc[0]
    range_new_up = tmp_df['new-upper'].iloc[0]
    range_new_low = tmp_df['new-lower'].iloc[0]

    # 菜单：罗列展示全部阈值选项（i.e.数据统计值如分位点等） <—— df_stats

    # 备选阈值数据准备
    tmp_df_stats = df_stats.loc[(df_stats['industry'] == this_industry) & (df_stats['factor'] == this_factor)]
    percentile_5 = tmp_df_stats['5%'].iloc[0]
    percentile_10 = tmp_df_stats['10%'].iloc[0]
    percentile_15 = tmp_df_stats['15%'].iloc[0]
    percentile_20 = tmp_df_stats['20%'].iloc[0]
    percentile_25 = tmp_df_stats['25%'].iloc[0]
    percentile_75 = tmp_df_stats['75%'].iloc[0]
    percentile_80 = tmp_df_stats['80%'].iloc[0]
    percentile_85 = tmp_df_stats['85%'].iloc[0]
    percentile_90 = tmp_df_stats['90%'].iloc[0]
    percentile_95 = tmp_df_stats['95%'].iloc[0]

    sigma_2_neg = tmp_df_stats['-2'].iloc[0]
    sigma_175_neg = tmp_df_stats['-1.75'].iloc[0]
    sigma_15_neg = tmp_df_stats['-1.5'].iloc[0]
    sigma_125_neg = tmp_df_stats['-1.25'].iloc[0]
    sigma_1_neg = tmp_df_stats['-1'].iloc[0]
    sigma_1 = tmp_df_stats['1'].iloc[0]
    sigma_125 = tmp_df_stats['1.25'].iloc[0]
    sigma_15 = tmp_df_stats['1.5'].iloc[0]
    sigma_175 = tmp_df_stats['1.75'].iloc[0]
    sigma_2 = tmp_df_stats['2'].iloc[0]

    old_lower = df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor), 'old-lower'].iloc[0]
    old_upper = df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor), 'old-upper'].iloc[0]
    new_lower = df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor), 'new-lower'].iloc[0]
    new_upper = df.loc[(df['industry'] == this_industry) & (df['factor'] == origin_this_factor), 'new-upper'].iloc[0]

    m_25_upper = tmp_df_stats['m=2.5:上限'].iloc[0]
    m_3_upper = tmp_df_stats['m=3:上限'].iloc[0]
    m_35_upper = tmp_df_stats['m=3.5:上限'].iloc[0]
    m_4_upper = tmp_df_stats['m=4:上限'].iloc[0]
    m_45_upper = tmp_df_stats['m=4.5:上限'].iloc[0]
    m_5_upper = tmp_df_stats['m=5:上限'].iloc[0]
    m_55_upper = tmp_df_stats['m=5.5:上限'].iloc[0]
    m_6_upper = tmp_df_stats['m=6:上限'].iloc[0]
    m_65_upper = tmp_df_stats['m=6.5:上限'].iloc[0]
    m_7_upper = tmp_df_stats['m=7:上限'].iloc[0]
    m_75_upper = tmp_df_stats['m=7.5:上限'].iloc[0]
    m_8_upper = tmp_df_stats['m=8:上限'].iloc[0]

    m_25_lower = tmp_df_stats['m=2.5:下限'].iloc[0]
    m_3_lower = tmp_df_stats['m=3:下限'].iloc[0]
    m_35_lower = tmp_df_stats['m=3.5:下限'].iloc[0]
    m_4_lower = tmp_df_stats['m=4:下限'].iloc[0]
    m_45_lower = tmp_df_stats['m=4.5:下限'].iloc[0]
    m_5_lower = tmp_df_stats['m=5:下限'].iloc[0]
    m_55_lower = tmp_df_stats['m=5.5:下限'].iloc[0]
    m_6_lower = tmp_df_stats['m=6:下限'].iloc[0]
    m_65_lower = tmp_df_stats['m=6.5:下限'].iloc[0]
    m_7_lower = tmp_df_stats['m=7:下限'].iloc[0]
    m_75_lower = tmp_df_stats['m=7.5:下限'].iloc[0]
    m_8_lower = tmp_df_stats['m=8:下限'].iloc[0]

    graph_figures(this_industry, origin_this_factor, 1, range_low, range_up, range_new_low, range_new_up)
    ui.select.refresh_png(this_path_1)

    if this_direction == '向上':
        my_choice_list = list()
        my_choice_list.append(f"1. 老阈值上限： {old_upper}")
        my_choice_list.append(f"2. 新阈值上限： {new_upper}")
        my_choice_list.append('   ')
        my_choice_list.append(f"3. 75百分位数： {percentile_75}")
        my_choice_list.append(f"4. 80百分位数： {percentile_80}")
        my_choice_list.append(f"5. 85百分位数： {percentile_85}")
        my_choice_list.append(f"6. 90百分位数： {percentile_90}")
        my_choice_list.append(f"7. 95百分位数： {percentile_95}")
        my_choice_list.append('   ')
        my_choice_list.append(f"8. 1标准差：      {sigma_1}")
        my_choice_list.append(f"9. 1.25标准差：  {sigma_125}")
        my_choice_list.append(f"10. 1.5标准差：  {sigma_15}")
        my_choice_list.append(f"11. 1.75标准差：{sigma_175}")
        my_choice_list.append(f"12. 2标准差：    {sigma_2}")
        my_choice_list.append('   ')
        my_choice_list.append(f"13. m=2.5上限：{m_25_upper}")
        my_choice_list.append(f"14. m=3上限：   {m_3_upper}")
        my_choice_list.append(f"15. m=3.5上限：{m_35_upper}")
        my_choice_list.append(f"16. m=4上限：   {m_4_upper}")
        my_choice_list.append(f"17. m=4.5上限：{m_45_upper}")
        my_choice_list.append(f"18. m=5上限：   {m_5_upper}")
        my_choice_list.append(f"19. m=5.5上限：{m_55_upper}")
        my_choice_list.append(f"20. m=6上限：   {m_6_upper}")
        my_choice_list.append(f"21. m=6.5上限：{m_65_upper}")
        my_choice_list.append(f"22. m=7上限：   {m_7_upper}")
        my_choice_list.append(f"23. m=7.5上限：{m_75_upper}")
        my_choice_list.append(f"24. m=8上限：   {m_8_upper}")
        ui.select.listView.addItems(my_choice_list)

        print("请选择阈值上限：")

    if this_direction == '向下':
        my_choice_list = list()
        my_choice_list.append(f"a. 老阈值下限：{old_lower}")
        my_choice_list.append(f"b. 新阈值下限：{new_lower}")
        my_choice_list.append('   ')
        my_choice_list.append(f"c. 5百分位数： {percentile_5}")
        my_choice_list.append(f"d. 10百分位数：{percentile_10}")
        my_choice_list.append(f"e. 15百分位数：{percentile_15}")
        my_choice_list.append(f"f. 20百分位数： {percentile_20}")
        my_choice_list.append(f"g. 25百分位数：{percentile_25}")
        my_choice_list.append('   ')
        my_choice_list.append(f"h. -1标准差：     {sigma_1_neg}")
        my_choice_list.append(f"i. -1.25标准差： {sigma_125_neg}")
        my_choice_list.append(f"j. -1.5标准差：   {sigma_15_neg}")
        my_choice_list.append(f"k. -1.75标准差：{sigma_175_neg}")
        my_choice_list.append(f"l. -2标准差：     {sigma_2_neg}")
        my_choice_list.append('   ')
        my_choice_list.append(f"m. m=2.5下限：{m_25_lower}")
        my_choice_list.append(f"n. m=3下限：   {m_3_lower}")
        my_choice_list.append(f"o. m=3.5下限：{m_35_lower}")
        my_choice_list.append(f"p. m=4下限：   {m_4_lower}")
        my_choice_list.append(f"q. m=4.5下限：{m_45_lower}")
        my_choice_list.append(f"r. m=5下限：   {m_5_lower}")
        my_choice_list.append(f"s. m=5.5下限：{m_55_lower}")
        my_choice_list.append(f"t. m=6下限：   {m_6_lower}")
        my_choice_list.append(f"u. m=6.5下限：{m_65_lower}")
        my_choice_list.append(f"v. m=7下限：   {m_7_lower}")
        my_choice_list.append(f"w. m=7.5下限：{m_75_lower}")
        my_choice_list.append(f"x. m=8下限：   {m_8_lower}")
        ui.select.listView.addItems(my_choice_list)

        print("请选择的阈值下限：")

    if this_direction == '双向':
        my_choice_list = list()
        my_choice_list.append(f"1. 老阈值上限： {old_upper}")
        my_choice_list.append(f"2. 新阈值上限： {new_upper}")
        my_choice_list.append('   ')
        my_choice_list.append(f"3. 75百分位数： {percentile_75}")
        my_choice_list.append(f"4. 80百分位数： {percentile_80}")
        my_choice_list.append(f"5. 85百分位数： {percentile_85}")
        my_choice_list.append(f"6. 90百分位数： {percentile_90}")
        my_choice_list.append(f"7. 95百分位数：{percentile_95}")
        my_choice_list.append('   ')
        my_choice_list.append(f"8. 1标准差：      {sigma_1}")
        my_choice_list.append(f"9. 1.25标准差：  {sigma_125}")
        my_choice_list.append(f"10. 1.5标准差：   {sigma_15}")
        my_choice_list.append(f"11. 1.75标准差：{sigma_175}")
        my_choice_list.append(f"12. 2标准差：     {sigma_2}")
        my_choice_list.append('   ')
        my_choice_list.append(f"13. m=2.5上限：{m_25_upper}")
        my_choice_list.append(f"14. m=3上限：   {m_3_upper}")
        my_choice_list.append(f"15. m=3.5上限：{m_35_upper}")
        my_choice_list.append(f"16. m=4上限：   {m_4_upper}")
        my_choice_list.append(f"17. m=4.5上限：{m_45_upper}")
        my_choice_list.append(f"18. m=5上限：   {m_5_upper}")
        my_choice_list.append(f"19. m=5.5上限：{m_55_upper}")
        my_choice_list.append(f"20. m=6上限：   {m_6_upper}")
        my_choice_list.append(f"21. m=6.5上限：{m_65_upper}")
        my_choice_list.append(f"22. m=7上限：   {m_7_upper}")
        my_choice_list.append(f"23. m=7.5上限：{m_75_upper}")
        my_choice_list.append(f"24. m=8上限：   {m_8_upper}")

        my_choice_list.append('   ')
        my_choice_list.append(f"a. 老阈值下限： {old_lower}")
        my_choice_list.append(f"b. 新阈值下限： {new_lower}")
        my_choice_list.append('   ')
        my_choice_list.append(f"c. 5百分位数：  {percentile_5}")
        my_choice_list.append(f"d. 10百分位数： {percentile_10}")
        my_choice_list.append(f"e. 15百分位数： {percentile_15}")
        my_choice_list.append(f"f. 20百分位数： {percentile_20}")
        my_choice_list.append(f"g. 25百分位数： {percentile_25}")
        my_choice_list.append('   ')
        my_choice_list.append(f"h. -1标准差：     {sigma_1_neg}")
        my_choice_list.append(f"i. -1.25标准差： {sigma_125_neg}")
        my_choice_list.append(f"j. -1.5标准差：   {sigma_15_neg}")
        my_choice_list.append(f"k. -1.75标准差：{sigma_175_neg}")
        my_choice_list.append(f"l. -2标准差：     {sigma_2_neg}")
        my_choice_list.append('   ')
        my_choice_list.append(f"m. m=2.5下限：{m_25_lower}")
        my_choice_list.append(f"n. m=3下限：   {m_3_lower}")
        my_choice_list.append(f"o. m=3.5下限：{m_35_lower}")
        my_choice_list.append(f"p. m=4下限：   {m_4_lower}")
        my_choice_list.append(f"q. m=4.5下限：{m_45_lower}")
        my_choice_list.append(f"r. m=5下限：   {m_5_lower}")
        my_choice_list.append(f"s. m=5.5下限：{m_55_lower}")
        my_choice_list.append(f"t. m=6下限：   {m_6_lower}")
        my_choice_list.append(f"u. m=6.5下限：{m_65_lower}")
        my_choice_list.append(f"v. m=7下限：   {m_7_lower}")
        my_choice_list.append(f"w. m=7.5下限：{m_75_lower}")
        my_choice_list.append(f"x. m=8下限：   {m_8_lower}")

        ui.select.listView.addItems(my_choice_list)

        print("请分别选择的阈值上限和下限：")

    this_series = df_combine[df_combine['PROFILE_industry'] == this_industry][this_factor]
    this_anomaly_proportion = get_anomaly_rate(this_series, this_direction, range_low, range_up)
    ui.select.current_anomaly_display.setText('  {:.5%}'.format(this_anomaly_proportion))


def refresh_figure():
    """生成人工阈值预览图"""
    industry, factor, level = selected_list_comb[ui.select.num]
    this_key = ui.select.chose_range[0:2].replace('.', '')
    if this_key == '1':
        ui.select.artificial_upper = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[3:8]
        ui.select.chose_parameter = '-'
    elif this_key == '2':
        ui.select.artificial_upper = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[3:6]
        ui.select.chose_parameter = '-'
    elif this_key == '3':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == '4':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == '5':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == '6':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == '7':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == '8':
        ui.select.artificial_upper = float(ui.select.chose_range[8:])
        ui.select.chose_method = ui.select.chose_range[4:7]
        ui.select.chose_parameter = float(ui.select.chose_range[3:4])
    elif this_key == '9':
        ui.select.artificial_upper = float(ui.select.chose_range[11:])
        ui.select.chose_method = ui.select.chose_range[7:10]
        ui.select.chose_parameter = float(ui.select.chose_range[3:7])
    elif this_key == '10':
        ui.select.artificial_upper = float(ui.select.chose_range[11:])
        ui.select.chose_method = ui.select.chose_range[7:10]
        ui.select.chose_parameter = float(ui.select.chose_range[4:7])
    elif this_key == '11':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = ui.select.chose_range[8:11]
        ui.select.chose_parameter = float(ui.select.chose_range[4:8])
    elif this_key == '12':
        ui.select.artificial_upper = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = float(ui.select.chose_range[4:5])
    elif this_key == '13':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '14':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])
    elif this_key == '15':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '16':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])
    elif this_key == '17':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '18':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])
    elif this_key == '19':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '20':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])
    elif this_key == '21':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '22':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])
    elif this_key == '23':
        ui.select.artificial_upper = float(ui.select.chose_range[12:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:9])
    elif this_key == '24':
        ui.select.artificial_upper = float(ui.select.chose_range[10:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[6:7])

    elif this_key == 'a':
        if ui.select.chose_range[10:] == '0.01且>0':
            ui.select.artificial_lower = '0.01且>0'
        elif ui.select.chose_range[9:] == '0.005且>0':
            ui.select.artificial_lower = '0.005且>0'
        else:
            ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[3:8]
        ui.select.chose_parameter = '-'
    elif this_key == 'b':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[3:6]
        ui.select.chose_parameter = '-'
    elif this_key == 'c':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[4:7]
        ui.select.chose_parameter = int(ui.select.chose_range[3:4])
    elif this_key == 'd':
        ui.select.artificial_lower = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == 'e':
        ui.select.artificial_lower = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == 'f':
        ui.select.artificial_lower = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == 'g':
        ui.select.artificial_lower = float(ui.select.chose_range[10:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = int(ui.select.chose_range[3:5])
    elif this_key == 'h':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = float(ui.select.chose_range[3:5])
    elif this_key == 'i':
        ui.select.artificial_lower = float(ui.select.chose_range[12:])
        ui.select.chose_method = ui.select.chose_range[8:11]
        ui.select.chose_parameter = float(ui.select.chose_range[3:8])
    elif this_key == 'j':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = ui.select.chose_range[7:10]
        ui.select.chose_parameter = float(ui.select.chose_range[3:7])
    elif this_key == 'k':
        ui.select.artificial_lower = float(ui.select.chose_range[12:])
        ui.select.chose_method = ui.select.chose_range[8:11]
        ui.select.chose_parameter = float(ui.select.chose_range[3:8])
    elif this_key == 'l':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = ui.select.chose_range[5:8]
        ui.select.chose_parameter = float(ui.select.chose_range[3:5])
    elif this_key == 'm':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 'n':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])
    elif this_key == 'o':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 'p':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])
    elif this_key == 'q':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 'r':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])
    elif this_key == 's':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 't':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])
    elif this_key == 'u':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 'v':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])
    elif this_key == 'w':
        ui.select.artificial_lower = float(ui.select.chose_range[11:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:8])
    elif this_key == 'x':
        ui.select.artificial_lower = float(ui.select.chose_range[9:])
        ui.select.chose_method = 'm值边界点'
        ui.select.chose_parameter = float(ui.select.chose_range[5:6])

    origin_factor, factor = alter_factor(factor)
    graph_figures(industry, origin_factor, 2, np.nan, np.nan, ui.select.artificial_lower, ui.select.artificial_upper)
    ui.select.refresh_png(this_path_2)

    this_series = df_combine[df_combine['PROFILE_industry'] == industry][factor]
    this_direction = dff.loc[(dff['industry'] == industry) & (dff['factor'] == origin_factor), 'direction'].iloc[0]
    this_anomaly_proportion = get_anomaly_rate(this_series, this_direction, ui.select.artificial_lower, ui.select.artificial_upper)
    ui.select.current_anomaly_display.setText('  {:.5%}'.format(this_anomaly_proportion))


def output_layout_conversion(this_df):
    # 参数修改了，这里也要改（优化方法：函数间参数传递）
    norm_parameter_1 = Py1.parameter_dict[1][0]
    norm_parameter_2 = Py1.parameter_dict[2][0]
    lof_parameter_1 = Py1.parameter_dict[1][1]
    lof_parameter_2 = Py1.parameter_dict[2][1]

    this_df.loc[(this_df['type'] == "统计") & (this_df['method'] == '正态分布') & (this_df['level'] == '普通'), 'parameters'] = norm_parameter_1
    this_df.loc[(this_df['type'] == "统计") & (this_df['method'] == '正态分布') & (this_df['level'] == '高危'), 'parameters'] = norm_parameter_2
    this_df.loc[(this_df['type'] == "统计") & (this_df['method'] == 'LOF') & (this_df['level'] == '普通'), 'parameters'] = lof_parameter_1
    this_df.loc[(this_df['type'] == "统计") & (this_df['method'] == 'LOF') & (this_df['level'] == '高危'), 'parameters'] = lof_parameter_2
    this_df.loc[this_df['type'] == "-", 'parameters'] = '-'
    this_df.loc[this_df['type'] == "-", 'method'] = '-'

    this_df.loc[this_df['type'] == "专家", 'parameters'] = '-'
    this_df.loc[this_df['type'] == "专家", 'method'] = '-'
    this_df = this_df[['industry', 'factor', 'level', 'direction', 'mark', 'new-type', 'method', 'parameters', 'lower', 'upper']]
    return this_df


def next_item():
    """点击下一项时，储存当前结果，并转到下一项指标"""
    if ui.select.num == -1:
        ui.select.num += 1
        industry, factor, level = selected_list_comb[ui.select.num]
        ui.select.spinBox.setValue(ui.select.num)
        print('=' * 50)
        print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
        get_new_range(industry, factor, level)

    elif -1 < ui.select.num < len(selected_list_comb) - 1:

        industry, factor, level = selected_list_comb[ui.select.num]
        df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'lower'] = ui.select.artificial_lower
        df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'upper'] = ui.select.artificial_upper
        if ui.select.chose_method == '新阈值':
            pass
        else:
            df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'parameters'] = ui.select.chose_parameter
            df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'method'] = ui.select.chose_method
        try:
            if not np.isnan(ui.select.artificial_lower) or not np.isnan(ui.select.artificial_upper):
                markItem = QTableWidgetItem()
                markItem.setBackground(QBrush(QColor(154, 205, 50)))
                ui.contact.tableWidget.setItem(ui.select.num, 4, markItem)
        except:
            if ui.select.artificial_lower == '0.01且>0' or ui.select.artificial_lower == '0.005且>0':
                markItem = QTableWidgetItem()
                markItem.setBackground(QBrush(QColor(154, 205, 50)))
                ui.contact.tableWidget.setItem(ui.select.num, 4, markItem)
            else:
                raise Warning('阈值选取后，结果跳转失败！')

        ui.select.num += 1
        industry, factor, level = selected_list_comb[ui.select.num]
        ui.select.spinBox.setValue(ui.select.num)
        print('=' * 50)
        print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
        get_new_range(industry, factor, level)

    elif ui.select.num >= len(selected_list_comb) - 1:
        # 结束人工判断流程操作
        ui.select.num += 1
        ui.select.current_anomaly_display.setText('诶，没了？！做完啦！！！')
        ui.select.mean_display.setText("-")
        ui.select.median_display.setText('-')
        ui.select.var_display.setText('-')
        ui.select.direction_display.setText('-')
        ui.select.level_display.setText('-')
        ui.select.old_upper_display.setText('-')
        ui.select.old_lower_display.setText('-')
        ui.select.old_anomaly_display.setText('-')
        ui.select.new_upper_display.setText('-')
        ui.select.new_lower_display.setText('-')
        ui.select.new_anomaly_display.setText('-')
        ui.select.current_industry_display.setText('-')
        ui.select.note_display.setText('-')
        ui.select.listView.clear()
        ui.select.graph_display.clear()
        ui.select.current_industry_display.setText('正在保存结果……请勿进行操作')
        output_df = output_layout_conversion(df)
        output_df.to_excel(root_path + '/output/阈值判定结果.xlsx')
        ui.select.current_industry_display.setText('保存结果成功')


def last_item():
    """点击上一项时，转到上一项指标"""
    if ui.select.num > 0:
        ui.select.num -= 1
        industry, factor, level = selected_list_comb[ui.select.num]
        ui.select.spinBox.setValue(ui.select.num)
        print('=' * 50)
        print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
        get_new_range(industry, factor, level)


def go_to_page():
    """点击转到时，转到spinbox中的数字的指标"""
    ui.select.num = ui.select.spinBox.value()

    industry, factor, level = selected_list_comb[ui.select.num]
    print('=' * 50)
    print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
    get_new_range(industry, factor, level)


def view_this_result():
    """先保存现在的取值，写入表格，然后读取表格，画图"""

    industry, factor, level = selected_list_comb[ui.select.num]
    df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'lower'] = ui.select.artificial_lower
    df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'upper'] = ui.select.artificial_upper
    if ui.select.chose_method == '新阈值':
        pass
    else:
        df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'parameters'] = ui.select.chose_parameter
        df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'method'] = ui.select.chose_method
    try:
        if not np.isnan(ui.select.artificial_lower) or not np.isnan(ui.select.artificial_upper):
            markItem = QTableWidgetItem()
            markItem.setBackground(QBrush(QColor(154, 205, 50)))
            ui.contact.tableWidget.setItem(ui.select.num, 4, markItem)
    except:
        if ui.select.artificial_lower == '0.01且>0' or ui.select.artificial_lower == '0.005且>0':
            markItem = QTableWidgetItem()
            markItem.setBackground(QBrush(QColor(154, 205, 50)))
            ui.contact.tableWidget.setItem(ui.select.num, 4, markItem)
        else:
            raise Warning('阈值选取后，结果跳转失败！')

    this_lower = df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'lower'].iloc[0]
    this_upper = df.loc[(df['industry'] == industry) & (df['factor'] == factor) & (df['level'] == level), 'upper'].iloc[0]
    range_low = ui.select.old_lower_display.text()[2:]
    range_up = ui.select.old_upper_display.text()[2:]
    # range_low = dff.loc[(dff['industry'] == industry) & (dff['factor'] == factor), 'old-lower'].iloc[0]
    # range_up = dff.loc[(dff['industry'] == industry) & (dff['factor'] == factor), 'old-upper'].iloc[0]
    print('查看当前已储存阈值:', '当前下限为', this_lower, ' ;当前上限为', this_upper)
    graph_figures(industry, factor, 3, range_low, range_up, this_lower, this_upper)
    ui.select.refresh_png(this_path_3)
    origin_factor, factor = alter_factor(factor)
    this_series = df_combine[df_combine['PROFILE_industry'] == industry][factor]
    this_direction = dff.loc[(dff['industry'] == industry) & (dff['factor'] == origin_factor), 'direction'].iloc[0]
    this_anomaly_proportion = get_anomaly_rate(this_series, this_direction, this_lower, this_upper)
    ui.select.current_anomaly_display.setText('  {:.5%}'.format(this_anomaly_proportion))


def save_tmp_result():
    ui.select.mean_display.setText("-")
    ui.select.median_display.setText('-')
    ui.select.var_display.setText('-')
    ui.select.direction_display.setText('-')
    ui.select.level_display.setText('-')
    ui.select.old_upper_display.setText('-')
    ui.select.old_lower_display.setText('-')
    ui.select.old_anomaly_display.setText('-')
    ui.select.new_upper_display.setText('-')
    ui.select.new_lower_display.setText('-')
    ui.select.new_anomaly_display.setText('-')
    ui.select.current_industry_display.setText('-')
    ui.select.current_anomaly.setText('-')
    ui.select.note_display.setText('-')
    ui.select.listView.clear()
    filename = f"tmp_阈值判定结果{time.strftime('%Y%m%d-%H%M%S')}.xlsx"
    output_df = output_layout_conversion(df)
    output_df.to_excel(root_path + '/output/' + filename)
    ui.select.current_industry_display.setText('保存结果成功')
    ui.select.current_factor_display.setText(filename)


def show_mean():
    if ui.select.num == -1:
        print('无对象数据!')
    else:
        # 这里有点蠢，因为不知道怎么删除axvline的划线
        graph_figures(ui.select.current_industry_display.text(), ui.select.current_factor_display.text(), 3, np.nan,
                      np.nan)
        if ui.select.status_mean == 0:
            ax = plt.gca()
            l3 = ax.axvline(ui.select.mean_display.text()[2:], color='goldenrod', ls='--')
            plt.legend([l3], ['mean'])
            plt.savefig(this_path_4)
            ui.select.refresh_png(this_path_4)
            ui.select.status_mean = 1
        elif ui.select.status_mean == 1:
            ui.select.refresh_png(this_path_1)
            ui.select.status_mean = 0


def show_median():
    if ui.select.num == -1:
        print('无对象数据!')
    else:
        graph_figures(ui.select.current_industry_display.text(), ui.select.current_factor_display.text(), 3, np.nan,
                      np.nan)
        if ui.select.status_median == 0:
            ax = plt.gca()
            l3 = ax.axvline(ui.select.median_display.text()[2:], color='deepskyblue', ls='-')
            plt.legend([l3], ['median'])
            plt.savefig(this_path_5)
            ui.select.refresh_png(this_path_5)
            ui.select.status_median = 1
        elif ui.select.status_median == 1:
            ui.select.refresh_png(this_path_1)
            ui.select.status_median = 0


def tree_select_item():
    ui.select.num = int(ui.contact.treeWidget.currentItem().text(1))
    industry, factor, level = selected_list_comb[ui.select.num]
    ui.select.spinBox.setValue(ui.select.num)
    print('=' * 50)
    print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
    get_new_range(industry, factor, level)
    ui.qsl.setCurrentIndex(0)


def table_select_item():
    ui.select.num = int(ui.contact.tableWidget.currentItem().row())
    industry, factor, level = selected_list_comb[ui.select.num]
    ui.select.spinBox.setValue(ui.select.num)
    print('=' * 50)
    print(industry + ', ' + factor + ', ' + level + ', ' + f'指标索引值:{ui.select.num}')
    get_new_range(industry, factor, level)
    ui.qsl.setCurrentIndex(0)


class This_Window(QWidget, manual_selection_page_1.Ui_manual_selection_page):
    # 人工选择页面
    def __init__(self):
        super().__init__()
        self.chose_range = ''
        self.chose_method = ''
        self.chose_parameter = 0
        self.num = -1
        self.artificial_lower = np.nan
        self.artificial_upper = np.nan
        self.status_mean = 0
        self.status_median = 0
        # self.resize(1029, 796)
        self.setupUi(self)

        # 设置对象与数据的交互内容

        # “转到”按钮的设置：页面跳转
        self.quick_link.clicked.connect(go_to_page)

        # 点击下一项，设置点击选项的反馈函数
        self.listView.itemClicked.connect(self.clicked)

        # 点击列表中选项将自动生成人工阈值图预览
        self.listView.itemClicked.connect(refresh_figure)

        # # 设置slide条
        # ui.horizontalSlider.setMinimum(0)
        # ui.horizontalSlider.valueChanged.connect(slide_valueChange)

        # 设置按下mean触发事件
        self.mean_label.clicked.connect(show_mean)

        # 设置按下median触发事件
        self.median_label.clicked.connect(show_median)

        # 前后项设置: 跳转，并且后项能够将当前的人工阈值（artificial）储存
        self.next_item.clicked.connect(next_item)
        self.last_item.clicked.connect(last_item)

        # “保存查看当前已储存的阈值选择”按钮设置
        self.view_result.clicked.connect(view_this_result)

        # "导出"按钮，将当前df表格导出至根目录
        self.save_result.clicked.connect(save_tmp_result)

    def clicked(self, index):
        print(self.listView.item(self.listView.row(index)).text())
        self.chose_range = self.listView.item(self.listView.row(index)).text()

        # QMessageBox.information(self, 'ListWidget', '您选择了：' + self.listView.item(self.listView.row(index)).text())

    def refresh_png(self, filepath):
        """从指定的路径中读取图片，并更新图片"""
        img = QImage(filepath)
        result = img.scaled(int(self.graph_display.height() * img.width() / img.height()), self.graph_display.height(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.graph_display.setPixmap(QPixmap.fromImage(result))
        self.graph_display.setAlignment(Qt.AlignCenter)


class FrameContactPage(QWidget, table_widget_1.Ui_Form):
    # 阈值概览页面
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.tableWidget.setHorizontalHeaderLabels(['Index', 'Industry', 'Factor', 'Level', 'Mark'])
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.horizontalHeader().setHighlightSections(True)
        self.tableWidget.verticalHeader().setVisible(False)
        # self.tableWidget.horizontalHeader().setVisible(False)
        # 当引入树状的时候
        # self.treeWidget.setColumnCount(3)
        # self.treeWidget.setHeaderLabels(['Factor', 'Index', 'Mark'])
        # self.treeWidget.setColumnWidth(0, 400)


# 网上示例继承的是QWidget, 我这里用的是QMainWindow
class MainWidget(QMainWindow, home_page_structure.Ui_MainWindow):
    """
    主窗口
    """
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 实例化一个堆叠布局
        self.qsl = QStackedLayout(self.frame)
        # 实例化分页面
        self.select = This_Window()
        self.contact = FrameContactPage()
        # 加入到布局中
        self.qsl.addWidget(self.select)
        self.qsl.addWidget(self.contact)
        # 控制函数
        self.controller()

    def controller(self):
        self.pushButton_0.clicked.connect(self.switch)
        self.pushButton_1.clicked.connect(self.switch)

    def switch(self):
        sender = self.sender().objectName()

        index = {
            "pushButton_0": 0,
            "pushButton_1": 1,
        }

        self.qsl.setCurrentIndex(index[sender])


# =======================================主程序=======================================================
# ===================================================================================================


if __name__ == '__main__':

    # 生成主窗口
    app = QApplication(sys.argv)
    # 实例化
    ui = MainWidget()

    # 获取需要人工修改的指标,同时往树状表单中插入指标
    list_comb = list(zip(dff['industry'].to_list(), dff['factor'].to_list(), dff['level'].to_list()))
    selected_list_comb = list()

    for selected_industry, selected_factor, selected_level in list_comb:
        if df.loc[(df['industry'] == selected_industry) & (df['factor'] == selected_factor) & (df['level'] == selected_level), 'new-type'].iloc[0] == "人工":
            selected_list_comb.append((selected_industry, selected_factor, selected_level))

    ui.contact.tableWidget.setRowCount(len(selected_list_comb))

    j = 0
    for select_item in selected_list_comb:
        newItem = QTableWidgetItem(str(j))
        newItem.setTextAlignment(Qt.AlignRight | Qt.AlignBottom)
        ui.contact.tableWidget.setItem(j, 0, newItem)
        for k in range(1, 4):
            newItem = QTableWidgetItem(select_item[k-1])
            newItem.setTextAlignment(Qt.AlignRight | Qt.AlignBottom)
            ui.contact.tableWidget.setItem(j, k, newItem)
        j += 1

    # 设置页面中spinbox的值域范围
    ui.select.spinBox.setRange(0, len(selected_list_comb))

    # # 设置双击节点切换选项操作
    # ui.contact.treeWidget.itemDoubleClicked.connect(tree_select_item)

    # 设置点击切换
    ui.contact.tableWidget.itemDoubleClicked.connect(table_select_item)

    # 展示窗口
    ui.show()
    sys.exit(app.exec_())

