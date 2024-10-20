import csv
import random
import numpy as np

# 设置正态分布的参数，可能需要根据实际情况调整这些值
mu = 2  # 假设截止时间的均值是1.5小时
sigma = 1  # 假设截止时间的标准差是0.5小时
num = 3 #生成多少组数据

# 读取data.csv文件
with open('./数据集/data.csv', 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)  # 将所有数据读取为列表形式
start_point_list = []
end_point_list = []
# 确保data.csv中有数据
if not data:
    print("data.csv is empty.")
else:
    # 生成Evs.csv文件
    with open('./数据集/EVs.csv', 'w', newline='') as evsfile:
        writer = csv.writer(evsfile)
        for i in range(num):
            # 尝试生成足够的数据行直到达到data.csv的行数，因为一些行可能会被舍弃
            generated_rows = 0
            start_point_list = []
            end_point_list = []
            while generated_rows < len(data):
                # 随机选择两个不同的点作为起点和终点
                start_data = list(filter(lambda x:x not in start_point_list,data))
                end_data = list(filter(lambda x:x not in end_point_list,data))
                start_point = random.choice(start_data)
                end_point = random.choice(end_data)
                # 确保起点和终点不同 且终点不重复
                while end_point == start_point:
                    end_point = random.choice(end_data)

                # 随机生成初始电量、电池容量和行驶能耗
                while True:
                    initial_charge = random.uniform(10, 50)
                    battery_capacity = random.randint(40, 100)
                    if battery_capacity > initial_charge:
                        break
                driving_energy_consumption = random.uniform(10, 30)

                # 生成正态分布的电动汽车截止时间，直到它在0和3之间
                deadline = np.random.normal(mu, sigma)
                if 0 < deadline <= 5:
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
                    start_point_list.append(start_point)
                    end_point_list.append(end_point)

print("生成完成")