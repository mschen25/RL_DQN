import csv
import random
import numpy as np

# 设置正态分布的参数，可能需要根据实际情况调整这些值
mu = 1.5  # 假设截止时间的均值是1.5小时
sigma = 0.5  # 假设截止时间的标准差是0.5小时

# 读取data.csv文件
with open('./数据集/data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)  # 将所有数据读取为列表形式

# 确保data.csv中有数据
if not data:
    print("data.csv is empty.")
else:
    # 生成Evs.csv文件
    with open('./数据集/EVs.csv', 'w', newline='') as evsfile:
        writer = csv.writer(evsfile)

        # 尝试生成足够的数据行直到达到data.csv的行数，因为一些行可能会被舍弃
        generated_rows = 0
        while generated_rows < len(data):
            # 随机选择两个不同的点作为起点和终点
            start_point = random.choice(data)
            end_point = random.choice(data)
            while end_point == start_point:  # 确保起点和终点不同
                end_point = random.choice(data)

            # 随机生成初始电量、电池容量和行驶能耗
            initial_charge = random.uniform(10, 50)
            battery_capacity = random.uniform(40, 100)
            driving_energy_consumption = random.uniform(10, 30)

            # 生成正态分布的电动汽车截止时间，直到它在0和3之间
            deadline = np.random.normal(mu, sigma)
            if 0 <= deadline <= 3:
                # 写入数据到Evs.csv
                writer.writerow([
                    start_point[0], start_point[1],  # 起点纬度和经度
                    end_point[0], end_point[1],      # 终点纬度和经度
                    initial_charge,                   # 初始电量
                    battery_capacity,                 # 电池容量
                    deadline,                         # 电动汽车截止时间
                    driving_energy_consumption        # 行驶能耗
                ])
                generated_rows += 1  # 仅当行被成功写入时才增加计数