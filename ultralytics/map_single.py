import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv('results.csv')  #替换成训练结果的csv路径

# 获取'metrics/mAP_0.5'列的数据
mAP_05_data = data['     metrics/mAP_0.5']

# 绘制曲线
plt.plot(mAP_05_data, label='Model-1', color='red', linewidth=1)

# 添加图例
plt.legend(loc='lower right')

# 添加标题和坐标轴标签
plt.xlabel('Epoch')
plt.ylabel('mAP_0.5(%)')
plt.title('mAP_0.5 Curve')

# 网格线
plt.grid(True)

# 保存图像到同目录下
plt.savefig('mAP_05_curve.png')
# plt.show()
