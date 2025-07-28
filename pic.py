import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 使用图中实际数据替换占位符（按图调整值）
Nf_bulk = np.array([210, 1080, 3000, 15100])       # 块体疲劳寿命(X轴)
strain_bulk = np.array([0.098, 0.036, 0.012, 0.0043]) # 块体应变(Y轴)

Nf_joint = np.array([300, 2500, 9500, 25000])      # 焊点疲劳寿命
strain_joint = np.array([0.075, 0.03, 0.02, 0.009]) # 焊点应变

plt.figure(figsize=(8, 6))
# 绘制散点图（按原图样式）
plt.scatter(Nf_bulk, strain_bulk, s=60, marker='s',
            color='black', label='Bulk')
plt.scatter(Nf_joint, strain_joint, s=60, marker='o',
            edgecolor='red', facecolor='red', linewidth=1.5, label='Solder Joints')  # 原图为空心红圈

# 块体拟合（黑色实线）
log_Nf_bulk = np.log10(Nf_bulk)
log_strain_bulk = np.log10(strain_bulk)
slope_bulk, intercept_bulk, r_value, p_value, std_err = stats.linregress(log_Nf_bulk, log_strain_bulk)
x_fit_bulk = np.array([10, 100000])
y_fit_bulk = 10**(intercept_bulk) * (x_fit_bulk)**(slope_bulk)  # 转换回原始坐标系

# 焊点拟合（红色虚线）
log_Nf_joint = np.log10(Nf_joint)
log_strain_joint = np.log10(strain_joint)
slope_joint, intercept_joint, r_value, p_value, std_err = stats.linregress(log_Nf_joint, log_strain_joint)
x_fit_joint = np.array([10, 100000])
y_fit_joint = 10**(intercept_joint) * (x_fit_joint)**(slope_joint)  # 幂函数形式

# 对数拟合（原图显示非线性关系）
# 块体拟合（实线）
plt.plot(x_fit_bulk, y_fit_bulk, 'k-', linewidth=1.5)
plt.plot(x_fit_joint, y_fit_joint, 'r--', linewidth=1.5)
# 设置双对数坐标
plt.xscale('log')
plt.yscale('log')
# 按原图设置刻度范围
plt.xlim(10, 100000)  # X轴从10到10000
plt.ylim(0.001, 1)  # Y轴从0.001到0.01 (扩展到可见范围)

# 设置坐标标签（原图样式）
plt.xlabel("Fatigue life [Nf]", fontsize=16, labelpad=10)
plt.ylabel("von Mises effective strain range", fontsize=16, labelpad=10)

# 设置刻度位置（与原图匹配）
plt.xticks([10, 100, 1000, 10000, 100000], ['10', '100', '1000', '10000', '100000'], fontsize=15)
plt.yticks([0.001, 0.01, 0.1, 1], ['0.001', '0.01', '0.1', '1'], fontsize=15)  # 注意欧洲小数分隔符

plt.grid(True, which="major", ls="-", alpha=0.2)

plt.title(label= '95.5Sn-4.0Ag-0.5Cu', fontsize=16, pad=15, weight='bold')
plt.legend(fontsize=16, loc='upper right')  # 原图图例在右上角
plt.tight_layout()

# 保存为出版级矢量图
plt.savefig('fatigue_plot.eps', format='eps', dpi=1200)
plt.savefig('fatigue_plot.pdf', format='pdf', dpi=1200)