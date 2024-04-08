import os
import time

time_start = time.time()  # 记录开始时间
os.system('python 生成_1_阈值计算.py')
time_end = time.time()  # 记录结束时间
time_sum_1 = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('第一段程序运行时间为:' + str(time_sum_1) + 's')

time_start = time.time()  # 记录开始时间
os.system('python 生成_2_阈值统计.py')
time_end = time.time()  # 记录结束时间
time_sum_2 = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('第二段程序运行时间为:' + str(time_sum_2) + 's')

time_start = time.time()  # 记录开始时间
os.system('python 生成_3_数据统计.py')
time_end = time.time()  # 记录结束时间
time_sum_3 = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('第三段程序运行时间为:' + str(time_sum_3) + 's')

time_start = time.time()  # 记录开始时间
os.system('python 生成_4_新阈值.py')
time_end = time.time()  # 记录结束时间
time_sum_4 = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print('第四段程序运行时间为:' + str(time_sum_4) + 's')

