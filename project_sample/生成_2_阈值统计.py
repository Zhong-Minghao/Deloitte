import pandas as pd
import numpy as np
from pandas import ExcelWriter
import 生成_1_阈值计算 as Py1

#
version = Py1.version
date = Py1.date
original_data_1 = Py1.original_data_1
original_data_2 = Py1.original_data_2
old_range_data = Py1.old_range_data

# 读取阈值
df = pd.read_excel(old_range_data, header=1)

# 处理阈值表
df['level'] = ''
df['mark'] = ''

df1 = df[['细分行业', '指标代码', 'level', '方向\n（单向/双向）', '类别\n（统计/专家）', 'mark',  '异常下限', '异常上限']].rename(columns={
    '细分行业': 'industry',
    '指标代码': 'factor',
    '方向\n（单向/双向）': 'direction',
    '类别\n（统计/专家）': 'type',
    '异常下限': 'old-lower',
    '异常上限': 'old-upper'
})
df1['level'] = '普通'

df2 = df[['细分行业', '指标代码', 'level', '方向\n（单向/双向）', '类别\n（统计/专家）.1', 'mark', '异常下限.1', '异常上限.1']].rename(columns={
    '细分行业': 'industry',
    '指标代码': 'factor',
    '方向\n（单向/双向）': 'direction',
    '类别\n（统计/专家）.1': 'type',
    '异常下限.1': 'old-lower',
    '异常上限.1': 'old-upper'
})
df2['level'] = '高危'

df = df1.append(df2)
del df1
del df2


# 读取原始数据
df1 = Py1.read_df(original_data_1)
df2 = Py1.read_df(original_data_2)

# 拉取行业和指标
industries = list(df['industry'].unique())
factors = list(df['factor'].unique())
levels = list(df['level'].unique())

# 初始化统计列
# df['old-num_triggered'] = np.nan
# df['old-num_sample'] = 0
df['old-perc_triggered'] = np.nan

# 开始统计
for this_industry in industries[0:]:
    print(f"=========={this_industry}==========")
    for this_level in levels[:]:
        for this_factor in factors[0:]:
            print(this_factor, this_level)

            # 处理指标名称
            tmp_factor = this_factor
            if '_01_' in tmp_factor:
                tmp_factor = tmp_factor.replace('_01_', '_')

            # 获取数据样本
            if this_factor in df1.columns:
                this_sample = df1.loc[df1['PROFILE_industry'] == this_industry, tmp_factor].to_list()
            if this_factor in df2.columns:
                this_sample = df2.loc[df2['PROFILE_industry'] == this_industry, tmp_factor].to_list()

            num_sample = len(this_sample)
            num_triggered = np.nan

            # 触发规则
            this_rule = {
                'direction': df.loc[
                    (df['industry'] == this_industry) & (df['factor'] == this_factor) & (df['level'] == this_level), 'direction'].iloc[
                    0],
                'type': df.loc[
                    (df['industry'] == this_industry) & (df['factor'] == this_factor) & (df['level'] == this_level), 'type'].iloc[0],
                'lower': df.loc[
                    (df['industry'] == this_industry) & (df['factor'] == this_factor) & (df['level'] == this_level), 'old-lower'].iloc[0],
                'upper': df.loc[
                    (df['industry'] == this_industry) & (df['factor'] == this_factor) & (df['level'] == this_level), 'old-upper'].iloc[0]
            }

            # 判定触发
            if this_rule['direction'] == "向上":
                if this_rule['upper'] != "-":
                    num_triggered = len([x for x in this_sample if this_rule['upper'] < x < this_rule['lower']])

            elif this_rule['direction'] == "向下":
                if this_rule['lower'] != "-":
                    if '且>0' in str(this_rule['lower']):
                        num_triggered = len([x for x in this_sample if 0 < x < float(this_rule['lower'].split('且')[0])])
                    else:
                        num_triggered = len([x for x in this_sample if this_rule['upper'] < x < this_rule['lower']])

            elif this_rule['direction'] == "双向":
                if this_rule['lower'] != "-":
                    if '且>0' in str(this_rule['lower']):
                        num_triggered = len([x for x in this_sample if x > this_rule['upper']
                                     or (0 < x < float(this_rule['lower'].split('且')[0]))])
                    else:
                        num_triggered = len([x for x in this_sample if x > this_rule['upper'] or x < this_rule['lower']])

            # 计算触发占比
            perc_triggered = num_triggered / num_sample

            # 记录统计
            #df.loc[
                #(df['industry'] == this_industry) & (
                            #df['factor'] == this_factor) & (df['level'] == this_level), 'old-num_triggered'] = num_triggered
            #df.loc[
                #(df['industry'] == this_industry) & (
                            #df['factor'] == this_factor) & (df['level'] == this_level), 'old-num_sample'] = num_sample
            df.loc[
                (df['industry'] == this_industry) & (
                            df['factor'] == this_factor) & (df['level'] == this_level), 'old-perc_triggered'] = perc_triggered

## 读取新阈值
dff = pd.read_excel(r"./output/[1] 阈值计算_v" + f"{version}_{date}.xlsx", header=0)

dff['level'] = ''

dff1 = dff[['industry', 'factor', 'direction', 'num_nan', 'num_zero', 'num_superfluous_zero', 'num_extreme',
            'num_effective', 'method', 'parameters', 'lower_1', 'upper_1', 'trigger_zero']].rename(columns={'lower_1': 'new-lower', 'upper_1': 'new-upper'})

dff1['level'] = '普通'

dff2 = dff[['industry', 'factor', 'direction', 'num_nan', 'num_zero', 'num_superfluous_zero', 'num_extreme',
            'num_effective', 'method', 'parameters', 'lower_2', 'upper_2', 'trigger_zero']].rename(columns={'lower_2': 'new-lower', 'upper_2': 'new-upper'})
dff2['level'] = '高危'

dff = dff1.append(dff2)
del dff1
del dff2

# 初始化统计列
# dff['new-num_triggered'] = np.nan
# dff['new-num_sample'] = 0
dff['new-perc_triggered'] = np.nan

# 开始统计
for this_industry in industries[0:]:
    print(f"=========={this_industry}==========")
    for this_level in levels[:]:
        for this_factor in factors[0:]:
            print(this_factor, this_level)

            # 处理指标名称
            tmp_factor = this_factor
            if '_01_' in tmp_factor:
                tmp_factor = tmp_factor.replace('_01_', '_')

            # 获取数据样本
            if this_factor in df1.columns:
                this_sample = df1.loc[df1['PROFILE_industry'] == this_industry, tmp_factor].to_list()
            if this_factor in df2.columns:
                this_sample = df2.loc[df2['PROFILE_industry'] == this_industry, tmp_factor].to_list()

            num_sample = len(this_sample)
            num_triggered = np.nan

            # 触发规则
            this_rule = {
                'direction': dff.loc[
                    (dff['industry'] == this_industry) & (dff['factor'] == this_factor) & (
                                dff['level'] == this_level), 'direction'].iloc[
                    0],
                #'type': dff.loc[
                    #(dff['industry'] == this_industry) & (dff['factor'] == this_factor) & (
                                #dff['level'] == this_level), 'type'].iloc[0],
                'lower': dff.loc[
                    (dff['industry'] == this_industry) & (dff['factor'] == this_factor) & (
                                dff['level'] == this_level), 'new-lower'].iloc[0],
                'upper': dff.loc[
                    (dff['industry'] == this_industry) & (dff['factor'] == this_factor) & (
                                dff['level'] == this_level), 'new-upper'].iloc[0]
            }

            # 判定触发
            if this_rule['direction'] == "向上":
                if this_rule['upper'] != "-":
                    num_triggered = len([x for x in this_sample if x > this_rule['upper']])

            elif this_rule['direction'] == "向下":
                if this_rule['lower'] != "-":
                    if '且>0' in str(this_rule['lower']):
                        num_triggered = len([x for x in this_sample if 0 < x < float(this_rule['lower'].split('且')[0])])
                    else:
                        num_triggered = len([x for x in this_sample if x < this_rule['lower']])

            elif this_rule['direction'] == "双向":
                if this_rule['lower'] != "-":
                    if '且>0' in str(this_rule['lower']):
                        num_triggered = len([x for x in this_sample if x > this_rule['upper']
                                             or (0 < x < float(this_rule['lower'].split('且')[0]))])
                    else:
                        num_triggered = len(
                            [x for x in this_sample if x > this_rule['upper'] or x < this_rule['lower']])

            # 计算触发占比
            perc_triggered = num_triggered / num_sample

            # 记录统计
            # dff.loc[
            # (dff['industry'] == this_industry) & (
            # dff['factor'] == this_factor) & (dff['level'] == this_level), 'new-num_triggered'] = num_triggered
            # dff.loc[
            # (dff['industry'] == this_industry) & (
            # dff['factor'] == this_factor) & (dff['level'] == this_level), 'new-num_sample'] = num_sample
            dff.loc[
                (dff['industry'] == this_industry) & (
                        dff['factor'] == this_factor) & (
                            dff['level'] == this_level), 'new-perc_triggered'] = perc_triggered

##
dfff = df.merge(dff, how='left', on=['industry', 'factor', 'level', 'direction'])

# 特殊处理标识
dfff['special_differentsign'] = np.nan
dfff['special_overtriggered'] = np.nan
dfff['special_bothbounds0'] = np.nan
dfff['special_toofewsample'] = np.nan

dfff['trigger_zero'] = dfff['trigger_zero'].apply(lambda x: True if x == 1 else np.nan)

dfff['tmp_old-lower'] = dfff['old-lower'].apply(lambda x: float(x.split('且')[0]) if type(x) == str and x != '-' else x)
dfff['tmp_old-lower'] = dfff['tmp_old-lower'].apply(lambda x: np.nan if x == '-' else float(x))
dfff['tmp_old-upper'] = dfff['old-upper'].apply(lambda x: np.nan if x == '-' else float(x))
dfff.loc[(dfff['direction'] == "向下") & (dfff['tmp_old-lower'] * dfff['new-lower'] < 0), 'special_differentsign'] = True
dfff.loc[(dfff['direction'] == "向上") & (dfff['tmp_old-upper'] * dfff['new-upper'] < 0), 'special_differentsign'] = True
dfff.loc[(dfff['direction'] == "双向") &
         ((dfff['tmp_old-lower'] * dfff['new-lower'] < 0) | (dfff['tmp_old-upper'] * dfff['new-upper'] < 0)), 'special_differentsign'] = True
dfff = dfff.drop(columns=['tmp_old-lower', 'tmp_old-upper'])

dfff.loc[(dfff['level'] == "普通") & (dfff['new-perc_triggered'] > 0.35), 'special_overtriggered'] = True
dfff.loc[(dfff['level'] == "高危") & (dfff['new-perc_triggered'] > 0.15), 'special_overtriggered'] = True

dfff.loc[(dfff['new-lower'] == 0) & (dfff['new-upper'] == 0), 'special_bothbounds0'] = True

dfff.loc[dfff['num_effective'] < 100, 'special_toofewsample'] = True

# 特殊处理
dfff.loc[dfff['trigger_zero'] == True, 'mark'] += "0值占比过高；"
dfff.loc[dfff['special_differentsign'] == True, 'mark'] += "新老阈值异号；"
dfff.loc[dfff['special_overtriggered'] == True, 'mark'] += "触发比例过高；"
dfff.loc[dfff['special_bothbounds0'] == True, 'mark'] += "新阈值均为0；"
dfff.loc[dfff['special_toofewsample'] == True, 'mark'] += "有效样本过小；"

dfff['mark'] = dfff['mark'].apply(lambda x: x[:-1] if x.endswith("；") else x)
dfff = dfff.drop(columns=[x for x in dfff.columns if x.startswith('special_')])

dfff['diff-perc_triggered'] = dfff['new-perc_triggered'] - dfff['old-perc_triggered']

# 指标名称修正
dfff['factor'] = dfff['factor'].apply(lambda x: x.replace('_01_', '_') + '_01' if '_01_' in x else x)

# 记录类型变更
dfff['new-type'] = dfff['type']
dfff.loc[(dfff['type'] != '-') & (dfff['mark'] != ''), 'new-type'] = '人工'

# 整理格式
dfff = dfff[[
    'industry', 'factor', 'level', 'direction',
    'num_nan', 'num_zero', 'num_superfluous_zero', 'num_extreme', 'num_effective',
    'type', 'mark', 'new-type',
    'method', 'parameters',
    'old-lower', 'old-upper', 'old-perc_triggered',
    'new-lower', 'new-upper', 'new-perc_triggered',
    'diff-perc_triggered'
]]


## 分行业统计触发占比
mydf = dfff[dfff['new-type'] == "统计"].copy()
#mydf = dfff[dfff['type'] == "统计"].copy()
mydf['num_all'] = mydf['num_nan'] + mydf['num_superfluous_zero'] + mydf['num_extreme'] + mydf['num_effective']
mydf['num_triggered'] = mydf['new-perc_triggered'] * mydf['num_all']
mydf['num_triggered'] = mydf['num_triggered'].round(0).astype(int)

mydff = mydf.groupby(['level', 'industry'])['new-perc_triggered'].mean().reset_index()
mydff1 = mydf.groupby(['level'])['num_triggered'].sum().reset_index()
mydff2 = mydf.groupby(['level'])['num_all'].sum().reset_index()
mydff1 = mydff1.merge(mydff2, how='left')
mydff1['new-perc_triggered'] = mydff1['num_triggered'] / mydff1['num_all']
mydff1['industry'] = "合计"
mydff1 = mydff1[['level', 'industry', 'new-perc_triggered']]
mydff = mydff.append(mydff1)

del mydff1
del mydff2

#
writer = ExcelWriter(r"./output/[2] 阈值统计_v" + f"{version}_{date}.xlsx")

dfff.to_excel(writer, '阈值测算底稿', index=False)
mydff.to_excel(writer, '触发占比', index=False)
writer.save()
writer.close()
