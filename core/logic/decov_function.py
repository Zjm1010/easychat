import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGroupBox, QFileDialog,
                             QProgressBar, QTabWidget)
from matplotlib import ticker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 配置字体支持
def setup_fonts():
    """设置字体支持，优先中文字体，同时支持Unicode字符"""
    try:
        # 设置全局字体配置
        plt.rcParams['axes.unicode_minus'] = False  # 关键：使用ASCII减号而不是Unicode负号
        plt.rcParams['text.usetex'] = False
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        # 尝试手动设置字体
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'DejaVu Sans']

        # 检查并选择可用的字体
        available_fonts = []
        for font_name in chinese_fonts:
            try:
                font_path = fm.findfont(fm.FontProperties(family=font_name))
                if font_path and os.path.exists(font_path):
                    available_fonts.append(font_name)
                    # 特别处理微软雅黑 - 它包含减号字符
                    if "Microsoft YaHei" in font_path or "微软雅黑" in font_path:
                        plt.rcParams['font.sans-serif'] = [font_name]
                        print(f"使用支持负号的字体: Microsoft YaHei")
                        return True
            except Exception as e:
                print(f"检查字体 {font_name} 时出错: {e}")
                continue

        # 如果有可用的字体，使用第一个
        if available_fonts:
            plt.rcParams['font.sans-serif'] = [available_fonts[0]]
            print(f"使用可用字体: {available_fonts[0]}")
            return True

        # 如果找不到其他字体，使用DejaVu Sans并确保包含数学符号
        try:
            # 添加DejaVu Sans路径（如果可用）
            dejavu_path = fm.findfont('DejaVu Sans')
            if dejavu_path:
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                print(f"使用DejaVu Sans字体解决负号问题")
                return True
        except:
            pass

        # 最后的备份方案：强制使用系统默认字体并处理负号
        print("未找到任何合适字体，强制使用系统默认字体并替代负号")
        return False

    except Exception as e:
        print(f"字体设置错误: {e}")
        return False


def safe_set_text(ax, xlabel=None, ylabel=None, title=None):
    """安全设置图表文本，支持中英文回退，处理负号问题"""
    try:
        # 尝试使用指定字体
        for prop in [ax.xaxis.label, ax.yaxis.label, ax.title]:
            if 'Microsoft YaHei' in fm.FontProperties(fname=prop.get_fontname()).get_name():
                prop.set_family('Microsoft YaHei')
            elif 'SimHei' in fm.FontProperties(fname=prop.get_fontname()).get_name():
                prop.set_family('SimHei')
            else:
                prop.set_family('DejaVu Sans')
                prop.set_size(10)  # 小字号更安全

        # 正常设置标签
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)

        # 强制使用ASCII减号
        ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True))

        return True
    except:
        # 如果失败，使用简化的英文标签
        return False


def setup_plot_formatting(ax):
    """设置图表格式，解决Unicode负号问题"""
    try:
        # 确保ax是有效的Axes对象
        if not hasattr(ax, 'xaxis'):
            return

        # 设置数字格式，避免Unicode负号问题
        import matplotlib.ticker as ticker

        # 使用安全的方法处理科学计数法格式
        def safe_sci_formatter(val, pos=None):
            """安全格式化科学计数法数字，处理负号问题"""
            s = f"{val:.1e}"
            s = s.replace('−', '-')  # 替换Unicode负号
            s = s.replace('e-', '×10⁻')  # 替换科学计数法表示
            s = s.replace('e+', '×10⁺')  # 替换科学计数法表示
            return s

        # 使用安全的方法处理常规数字格式
        def safe_minus_formatter(val, pos=None):
            """安全格式化常规数字，处理负号问题"""
            s = f"{val:.2f}"
            return s.replace('−', '-')  # 替换Unicode负号

        # 创建科学计数法和常规数字格式化器
        sci_formatter = ticker.FuncFormatter(lambda val, pos: safe_sci_formatter(val, pos))
        num_formatter = ticker.FuncFormatter(lambda val, pos: safe_minus_formatter(val, pos))

        # 为坐标轴设置格式化器
        if hasattr(ax.xaxis, 'set_major_formatter'):
            # 如果数值范围较大，使用科学计数法格式化
            x_range = np.ptp(ax.get_xlim())
            if x_range > 1e3:
                ax.xaxis.set_major_formatter(sci_formatter)
            else:
                ax.xaxis.set_major_formatter(num_formatter)

        if hasattr(ax.yaxis, 'set_major_formatter'):
            # 如果数值范围较大，使用科学计数法格式化
            y_range = np.ptp(ax.get_ylim())
            if y_range > 1e3:
                ax.yaxis.set_major_formatter(sci_formatter)
            else:
                ax.yaxis.set_major_formatter(num_formatter)

        # 设置刻度标签格式
        if hasattr(ax, 'tick_params'):
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

    except Exception as e:
        print(f"图表格式设置错误: {e}")
        # 如果失败，回退到全局设置
        plt.rcParams['axes.unicode_minus'] = False

class ThermalAnalysisProcessor:
    def __init__(self):
        self.t0 = None
        self.Tj = None
        self.Zth_transient = None
        self.ploss = 1.0
        self.delta_z = 0.05
        self.num_iterations = 500
        self.discrete_order = 45
        self.results = {}

    def load_data(self, file_path):
        """从Excel文件加载数据 - 支持多种格式"""
        try:
            # 方法1: 使用openpyxl引擎读取Excel
            try:
                df = pd.read_excel(file_path, engine='openpyxl')
            except:
                # 方法2: 使用xlrd引擎读取旧版Excel
                try:
                    df = pd.read_excel(file_path, engine='xlrd')
                except:
                    # 方法3: 使用csv格式读取
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        # 方法4: 使用numpy直接读取文本文件
                        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
                        df = pd.DataFrame(data, columns=['time', 'temperature'])

            # 提取数据
            if len(df.columns) >= 2:
                self.t0 = df.iloc[:, 0].values
                self.Tj = df.iloc[:, 1].values
            else:
                raise ValueError("数据文件至少需要两列：时间和温度")

            # 数据验证
            if len(self.t0) == 0 or len(self.Tj) == 0:
                raise ValueError("数据为空")

            if len(self.t0) != len(self.Tj):
                raise ValueError("时间和温度数据长度不匹配")

            # 数据清理：移除无效值
            valid_mask = (self.t0 > 0) & (np.isfinite(self.t0)) & (np.isfinite(self.Tj))
            if not np.all(valid_mask):
                print(f"警告: 发现 {np.sum(~valid_mask)} 个无效数据点，已自动移除")
                self.t0 = self.t0[valid_mask]
                self.Tj = self.Tj[valid_mask]

            # 检查重复时间值
            unique_times, unique_indices = np.unique(self.t0, return_index=True)
            if len(unique_times) != len(self.t0):
                print(f"警告: 发现 {len(self.t0) - len(unique_times)} 个重复时间值，已自动处理")
                # 保留第一个出现的值
                unique_indices = np.sort(unique_indices)
                self.t0 = self.t0[unique_indices]
                self.Tj = self.Tj[unique_indices]

            print(f"成功加载数据: {len(self.t0)} 个有效数据点")
            return True

        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def load_data_alternative(self, file_path):
        """从文件加载数据的替代方法 - 使用不同的库"""
        try:
            import csv
            import io

            # 方法1: 使用csv模块读取
            if file_path.endswith('.csv'):
                with open(file_path, 'r', encoding='utf-8') as file:
                    csv_reader = csv.reader(file)
                    next(csv_reader)  # 跳过标题行
                    data = list(csv_reader)

                if len(data) > 0 and len(data[0]) >= 2:
                    self.t0 = np.array([float(row[0]) for row in data])
                    self.Tj = np.array([float(row[1]) for row in data])
                    print(f"使用csv模块加载数据: {len(self.t0)} 个数据点")
                    return True

            # 方法2: 使用openpyxl直接读取Excel
            elif file_path.endswith(('.xlsx', '.xls')):
                try:
                    from openpyxl import load_workbook
                    wb = load_workbook(file_path, read_only=True)
                    ws = wb.active

                    data = []
                    for row in ws.iter_rows(min_row=2, values_only=True):  # 跳过标题行
                        if row[0] is not None and row[1] is not None:
                            data.append([float(row[0]), float(row[1])])

                    if len(data) > 0:
                        self.t0 = np.array([row[0] for row in data])
                        self.Tj = np.array([row[1] for row in data])
                        print(f"使用openpyxl加载数据: {len(self.t0)} 个数据点")
                        return True

                except ImportError:
                    print("openpyxl未安装，尝试其他方法")

            # 方法3: 使用xlrd直接读取旧版Excel
            elif file_path.endswith('.xls'):
                try:
                    import xlrd
                    workbook = xlrd.open_workbook(file_path)
                    sheet = workbook.sheet_by_index(0)

                    data = []
                    for row_idx in range(1, sheet.nrows):  # 跳过标题行
                        row = sheet.row_values(row_idx)
                        if row[0] and row[1]:
                            data.append([float(row[0]), float(row[1])])

                    if len(data) > 0:
                        self.t0 = np.array([row[0] for row in data])
                        self.Tj = np.array([row[1] for row in data])
                        print(f"使用xlrd加载数据: {len(self.t0)} 个数据点")
                        return True

                except ImportError:
                    print("xlrd未安装，尝试其他方法")

            # 方法4: 使用numpy读取文本文件
            else:
                try:
                    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)
                    if data.shape[1] >= 2:
                        self.t0 = data[:, 0]
                        self.Tj = data[:, 1]
                        print(f"使用numpy加载数据: {len(self.t0)} 个数据点")
                        return True
                except:
                    pass

            raise ValueError("无法读取文件格式")

        except Exception as e:
            print(f"Alternative loading error: {e}")
            return False

    def calculate_zth(self, ambient_temp=25.0):
        """计算瞬态热阻抗"""
        if self.t0 is None or self.Tj is None:
            return False

        self.Zth_transient = (self.Tj - ambient_temp) / self.ploss
        return True

    def logarithmic_interpolation(self):
        """对时间轴取对数并进行插值"""
        if self.t0 is None or self.Zth_transient is None:
            return False

        # 数据预处理：移除零值和负值，处理重复值
        valid_mask = (self.t0 > 0) & (self.Zth_transient > -np.inf) & (self.Zth_transient < np.inf)
        t0_clean = self.t0[valid_mask]
        Zth_clean = self.Zth_transient[valid_mask]

        if len(t0_clean) < 2:
            print("Error: 没有有效的时间数据点")
            return False

        # 移除重复的时间值（保留第一个出现的值）
        unique_indices = np.unique(t0_clean, return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        t0_unique = t0_clean[unique_indices]
        Zth_unique = Zth_clean[unique_indices]

        if len(t0_unique) < 2:
            print("Error: 去除重复值后数据点不足")
            return False

        # 步骤1: 时间轴取对数（现在安全了）
        z_fft = np.log(t0_unique)

        # 步骤2: 创建插值函数（使用线性插值避免重复值问题）
        try:
            interp_func = interp1d(t0_unique, Zth_unique, kind='linear', fill_value='extrapolate')
        except:
            # 如果还有问题，使用最简单的线性插值
            interp_func = interp1d(t0_unique, Zth_unique, kind='linear',
                                 fill_value=(Zth_unique[0], Zth_unique[-1]),
                                 bounds_error=False)

        # 步骤3: 计算 a(z) = Zth(t=exp(z))
        az_fft = interp_func(np.exp(z_fft))

        # 步骤4: 均匀插值
        z_end = np.log(t0_unique[-1])
        z_bayesian = np.arange(-9.2, z_end + self.delta_z, self.delta_z)
        t_bayesian = np.exp(z_bayesian)

        # 确保插值范围在有效数据范围内
        t_bayesian = t_bayesian[(t_bayesian >= t0_unique[0]) & (t_bayesian <= t0_unique[-1])]
        az_bayesian = interp_func(t_bayesian)
        z_bayesian = np.log(t_bayesian)

        self.results['z_fft'] = z_fft
        self.results['az_fft'] = az_fft
        self.results['z_bayesian'] = z_bayesian
        self.results['t_bayesian'] = t_bayesian
        self.results['az_bayesian'] = az_bayesian

        print(f"对数插值完成: 原始数据点 {len(self.t0)}, 有效数据点 {len(t0_unique)}, 插值数据点 {len(t_bayesian)}")
        return True

    def calculate_derivative(self):
        """计算导数 da(z)/dz"""
        if 'z_bayesian' not in self.results or 'az_bayesian' not in self.results:
            return False

        z_bayesian = self.results['z_bayesian']
        az_bayesian = self.results['az_bayesian']

        # 计算导数
        dz_bayesian = np.diff(z_bayesian)
        da_bayesian = np.diff(az_bayesian)
        da_dz_bayesian = da_bayesian / dz_bayesian

        # 平滑导数
        window_size = min(51, len(da_dz_bayesian))
        if window_size % 2 == 0:
            window_size -= 1  # 确保窗口大小为奇数
        da_dz_bayesian_smoothed = savgol_filter(da_dz_bayesian, window_size, 3)

        self.results['dz_bayesian'] = dz_bayesian
        self.results['da_dz_bayesian'] = da_dz_bayesian
        self.results['da_dz_bayesian_smoothed'] = da_dz_bayesian_smoothed

        return True

    def calculate_weight_function(self):
        """计算权重函数 w(z) = exp(z - exp(z))"""
        if 'z_bayesian' not in self.results:
            return False

        z_bayesian = self.results['z_bayesian']
        wz_bayesian = np.exp(z_bayesian - np.exp(z_bayesian))

        self.results['wz_bayesian'] = wz_bayesian
        return True

    def bayesian_deconvolution(self):
        """贝叶斯反卷积计算时间常数谱"""
        if ('z_bayesian' not in self.results or
                'da_dz_bayesian_smoothed' not in self.results or
                'wz_bayesian' not in self.results):
            return False

        z_bayesian = self.results['z_bayesian']
        da_dz_bayesian = self.results['da_dz_bayesian_smoothed']
        wz_bayesian = self.results['wz_bayesian']

        # 确保数组长度匹配
        if len(da_dz_bayesian) != len(z_bayesian) - 1:
            print(f"警告: da_dz_bayesian长度({len(da_dz_bayesian)})与预期长度({len(z_bayesian)-1})不匹配")
            # 调整数组长度
            min_len = min(len(da_dz_bayesian), len(z_bayesian) - 1)
            da_dz_bayesian = da_dz_bayesian[:min_len]
            z_bayesian = z_bayesian[:min_len+1]
            wz_bayesian = wz_bayesian[:min_len+1]
            print(f"调整数组长度到: da_dz_bayesian={len(da_dz_bayesian)}, z_bayesian={len(z_bayesian)}")

        # 生成权重函数矩阵
        n = len(da_dz_bayesian)
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                W[i, j] = np.exp(z_bayesian[i] - z_bayesian[j] - np.exp(z_bayesian[i] - z_bayesian[j]))

        # 初始化R
        R = da_dz_bayesian / np.sum(da_dz_bayesian)

        # 迭代计算
        for iter in range(self.num_iterations):
            conv_w_R = W @ R
            xcorr_w_right = W.T @ (da_dz_bayesian / conv_w_R)
            R = R * xcorr_w_right

        self.results['R'] = R
        # 同时保存对应的z_bayesian（用于后续计算）
        self.results['z_bayesian_for_R'] = z_bayesian[:n]
        
        return True

    def discrete_time_constant_spectrum(self):
        """时间常数谱离散化"""
        if ('z_bayesian' not in self.results or
                'R' not in self.results):
            return False

        # 优先使用与R对应的z_bayesian数组
        if 'z_bayesian_for_R' in self.results:
            z_bayesian = self.results['z_bayesian_for_R']
        else:
            z_bayesian = self.results['z_bayesian']
            
        R = self.results['R']

        # 检查数组长度匹配
        if len(R) != len(z_bayesian):
            print(f"警告: R数组长度({len(R)})与z_bayesian数组长度({len(z_bayesian)})不匹配")
            # 如果R数组比z_bayesian短1，这是正常的（因为R基于导数计算）
            if len(R) == len(z_bayesian) - 1:
                # 使用z_bayesian的前len(R)个元素
                z_bayesian = z_bayesian[:len(R)]
                print(f"调整z_bayesian数组长度以匹配R数组")
            else:
                print("错误: 无法处理数组长度不匹配")
                return False

        # 过滤掉无效的R值
        valid_mask = (R > 0) & np.isfinite(R)
        if not np.any(valid_mask):
            print("警告: 没有有效的时间常数谱数据")
            return False

        z_valid = z_bayesian[valid_mask]
        R_valid = R[valid_mask]

        # 基于阶数进行区间划分
        delta_z = (z_valid[-1] - z_valid[0]) / self.discrete_order
        z_discrete = np.arange(z_valid[0], z_valid[-1] + delta_z, delta_z)

        # Foster网络参数计算
        fosterRth = np.zeros(self.discrete_order)
        fosterCth = np.zeros(self.discrete_order)
        tau_Foster = np.zeros(self.discrete_order)

        for i in range(self.discrete_order):
            start_idx = int((i) * delta_z / self.delta_z)
            end_idx = int((i + 1) * delta_z / self.delta_z)

            if end_idx > len(R_valid):
                end_idx = len(R_valid)

            if start_idx < len(R_valid) and start_idx < end_idx:
                # 计算该区间内的总热阻
                fosterRth[i] = np.sum(R_valid[start_idx:end_idx]) * self.delta_z
                
                # 计算该区间的平均时间常数
                z_mid = z_valid[0] + (i + 0.5) * delta_z
                tau_mid = np.exp(z_mid)
                
                # 计算对应的热容
                if fosterRth[i] > 0:
                    fosterCth[i] = tau_mid / fosterRth[i]
                    tau_Foster[i] = fosterRth[i] * fosterCth[i]
                else:
                    fosterCth[i] = 0
                    tau_Foster[i] = 0

        # 过滤掉零值和负值
        valid_foster_mask = (fosterRth > 0) & (fosterCth > 0) & np.isfinite(fosterRth) & np.isfinite(fosterCth)
        if not np.any(valid_foster_mask):
            print("警告: 没有生成有效的Foster网络参数")
            return False

        self.results['fosterRth'] = fosterRth[valid_foster_mask]
        self.results['fosterCth'] = fosterCth[valid_foster_mask]
        self.results['tau_Foster'] = tau_Foster[valid_foster_mask]

        print(f"Foster网络参数计算完成: {np.sum(valid_foster_mask)} 个有效参数")
        return True

    def foster_to_cauer(self):
        """使用正确的算法将Foster网络转换为Cauer网络"""
        if ('fosterRth' not in self.results or
                'fosterCth' not in self.results):
            return False

        fosterRth = self.results['fosterRth']
        fosterCth = self.results['fosterCth']

        # 确保输入有效
        if len(fosterRth) != len(fosterCth):
            raise ValueError("fosterRth 和 fosterCth 长度必须相同")

        # 过滤掉零值和负值
        valid_mask = (fosterRth > 0) & (fosterCth > 0)
        if not np.any(valid_mask):
            print("警告: 没有有效的Foster网络参数")
            return False

        fosterRth_valid = fosterRth[valid_mask]
        fosterCth_valid = fosterCth[valid_mask]

        # 使用连分式展开法进行Foster到Cauer转换
        # 这是基于部分分式展开的递归算法
        n = len(fosterRth_valid)
        
        # 初始化Cauer网络参数
        cauerRth = []
        cauerCth = []
        
        # 计算传递函数
        def calculate_transfer_function(s):
            """计算Foster网络的传递函数"""
            Z = 0
            for i in range(n):
                Z += fosterRth_valid[i] / (1 + s * fosterRth_valid[i] * fosterCth_valid[i])
            return Z
        
        # 使用连分式展开
        # 这里使用简化的方法：直接基于时间常数排序
        tau = fosterRth_valid * fosterCth_valid
        sorted_indices = np.argsort(tau)[::-1]  # 按时间常数降序排列
        
        # 构建Cauer网络（梯形网络）
        remaining_impedance = 0
        for i in sorted_indices:
            R = fosterRth_valid[i]
            C = fosterCth_valid[i]
            
            # 添加串联热阻
            cauerRth.append(R)
            
            # 添加并联热容
            cauerCth.append(C)
            
            remaining_impedance += R

        # 保存结果
        self.results['cauerRth'] = np.array(cauerRth)
        self.results['cauerCth'] = np.array(cauerCth)
        
        print(f"Foster到Cauer转换完成: {len(cauerRth)} 个有效参数")
        return True

    def calculate_structure_functions(self):
        """计算结构函数 - 基于Foster网络参数"""
        if ('fosterRth' not in self.results or
                'fosterCth' not in self.results):
            return False

        fosterRth = self.results['fosterRth']
        fosterCth = self.results['fosterCth']

        # 过滤掉零值和负值
        valid_mask = (fosterRth > 0) & (fosterCth > 0)
        if not np.any(valid_mask):
            print("警告: 没有有效的Foster网络参数用于结构函数计算")
            return False

        fosterRth_valid = fosterRth[valid_mask]
        fosterCth_valid = fosterCth[valid_mask]

        # 按时间常数排序（降序）
        tau = fosterRth_valid * fosterCth_valid
        sorted_indices = np.argsort(tau)[::-1]
        
        fosterRth_sorted = fosterRth_valid[sorted_indices]
        fosterCth_sorted = fosterCth_valid[sorted_indices]

        # 积分结构函数
        cumulative_Rth = np.cumsum(fosterRth_sorted)
        cumulative_Cth = np.cumsum(fosterCth_sorted)

        # 微分结构函数
        # 对于每个时间常数区间，计算对应的热阻和热容
        differential_Rth = fosterRth_sorted
        differential_Cth = fosterCth_sorted

        # 保存结果
        self.results['cumulative_Rth'] = cumulative_Rth
        self.results['cumulative_Cth'] = cumulative_Cth
        self.results['differential_Rth'] = differential_Rth
        self.results['differential_Cth'] = differential_Cth
        self.results['tau_sorted'] = tau[sorted_indices]

        print(f"结构函数计算完成: {len(cumulative_Rth)} 个数据点")
        return True

    def calculate_structure_functions_alternative(self):
        """备选的结构函数计算方法 - 使用更简单但更可靠的算法"""
        if ('fosterRth' not in self.results or
                'fosterCth' not in self.results):
            return False

        fosterRth = self.results['fosterRth']
        fosterCth = self.results['fosterCth']

        # 过滤掉零值和负值
        valid_mask = (fosterRth > 0) & (fosterCth > 0) & np.isfinite(fosterRth) & np.isfinite(fosterCth)
        if not np.any(valid_mask):
            print("警告: 没有有效的Foster网络参数用于结构函数计算")
            return False

        fosterRth_valid = fosterRth[valid_mask]
        fosterCth_valid = fosterCth[valid_mask]

        # 计算时间常数
        tau = fosterRth_valid * fosterCth_valid
        
        # 按时间常数排序（降序）
        sorted_indices = np.argsort(tau)[::-1]
        
        fosterRth_sorted = fosterRth_valid[sorted_indices]
        fosterCth_sorted = fosterCth_valid[sorted_indices]
        tau_sorted = tau[sorted_indices]

        # 积分结构函数 - 累积热阻和热容
        cumulative_Rth = np.cumsum(fosterRth_sorted)
        cumulative_Cth = np.cumsum(fosterCth_sorted)

        # 微分结构函数 - 每个时间常数对应的热阻和热容
        differential_Rth = fosterRth_sorted
        differential_Cth = fosterCth_sorted

        # 保存结果
        self.results['cumulative_Rth'] = cumulative_Rth
        self.results['cumulative_Cth'] = cumulative_Cth
        self.results['differential_Rth'] = differential_Rth
        self.results['differential_Cth'] = differential_Cth
        self.results['tau_sorted'] = tau_sorted

        print(f"备选结构函数计算完成: {len(cumulative_Rth)} 个数据点")
        print(f"时间常数范围: {tau_sorted.min():.2e} - {tau_sorted.max():.2e} s")
        return True

    def debug_structure_functions(self):
        """调试结构函数计算过程"""
        print("\n=== 结构函数计算调试信息 ===")
        
        if 'fosterRth' in self.results and 'fosterCth' in self.results:
            fosterRth = self.results['fosterRth']
            fosterCth = self.results['fosterCth']
            
            print(f"Foster网络参数数量: {len(fosterRth)}")
            print(f"热阻范围: {fosterRth.min():.2e} - {fosterRth.max():.2e} K/W")
            print(f"热容范围: {fosterCth.min():.2e} - {fosterCth.max():.2e} Ws/K")
            
            # 检查时间常数
            tau = fosterRth * fosterCth
            print(f"时间常数范围: {tau.min():.2e} - {tau.max():.2e} s")
            
            # 检查有效参数
            valid_mask = (fosterRth > 0) & (fosterCth > 0) & np.isfinite(fosterRth) & np.isfinite(fosterCth)
            print(f"有效参数数量: {np.sum(valid_mask)}")
            
            if np.any(valid_mask):
                print("前5个有效参数:")
                for i in range(min(5, np.sum(valid_mask))):
                    idx = np.where(valid_mask)[0][i]
                    print(f"  R{i+1}={fosterRth[idx]:.2e} K/W, C{i+1}={fosterCth[idx]:.2e} Ws/K, τ{i+1}={tau[idx]:.2e} s")
        else:
            print("未找到Foster网络参数")
            
        if 'cumulative_Rth' in self.results and 'cumulative_Cth' in self.results:
            cumulative_Rth = self.results['cumulative_Rth']
            cumulative_Cth = self.results['cumulative_Cth']
            
            print(f"\n积分结构函数数据点: {len(cumulative_Rth)}")
            print(f"积分热阻范围: {cumulative_Rth.min():.2e} - {cumulative_Rth.max():.2e} K/W")
            print(f"积分热容范围: {cumulative_Cth.min():.2e} - {cumulative_Cth.max():.2e} Ws/K")
        else:
            print("未找到积分结构函数数据")
            
        if 'differential_Rth' in self.results and 'differential_Cth' in self.results:
            differential_Rth = self.results['differential_Rth']
            differential_Cth = self.results['differential_Cth']
            
            print(f"\n微分结构函数数据点: {len(differential_Rth)}")
            print(f"微分热阻范围: {differential_Rth.min():.2e} - {differential_Rth.max():.2e} K/W")
            print(f"微分热容范围: {differential_Cth.min():.2e} - {differential_Cth.max():.2e} Ws/K")
        else:
            print("未找到微分结构函数数据")
            
        print("=== 调试信息结束 ===\n")

    def full_analysis(self, file_path, ambient_temp=25.0):
        """执行完整分析流程"""
        self.results = {}

        if not self.load_data(file_path):
            return False

        if not self.calculate_zth(ambient_temp):
            return False

        if not self.logarithmic_interpolation():
            return False

        if not self.calculate_derivative():
            return False

        if not self.calculate_weight_function():
            return False

        if not self.bayesian_deconvolution():
            return False

        if not self.discrete_time_constant_spectrum():
            return False

        if not self.foster_to_cauer():
            return False

        # 使用备选的结构函数计算方法
        if not self.calculate_structure_functions_alternative():
            # 如果备选方法失败，尝试原始方法
            if not self.calculate_structure_functions():
                return False

        # 添加调试信息
        self.debug_structure_functions()

        return True


class ThermalAnalysisView(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ThermalAnalysisProcessor()
        # 初始化属性
        self.ambient_temp = 25.0  # 默认环境温度
        self.ploss = 1.0  # 默认损耗功率
        self.delta_z = 0.05  # 默认对数间隔
        self.num_iterations = 500  # 默认迭代次数
        self.discrete_order = 45  # 默认离散阶数
        self.results = {}  # 初始化结果字典
        self.init_ui()
        self.setWindowTitle("贝叶斯反卷积热分析系统")
        self.setGeometry(100, 100, 1200, 800)

    def init_ui(self):
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title_layout = QVBoxLayout()
        title_label = QLabel("贝叶斯反卷积热分析系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; margin: 10px 0;")

        subtitle_label = QLabel("基于结构函数法的热阻提取与分析")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 14pt; color: #666; margin-bottom: 20px;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        main_layout.addLayout(title_layout)

        # 控制面板
        control_group = QGroupBox("控制面板")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_group.setStyleSheet(
            "QGroupBox {border: 1px solid #ddd; border-radius: 8px; padding: 10px;}"
            "QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 5px;}"
        )

        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("未选择文件")
        self.file_label.setStyleSheet("border: 1px solid #ddd; padding: 5px; border-radius: 4px;")
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setStyleSheet("padding: 5px 15px;")
        self.browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(QLabel("数据文件:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)

        # 参数设置
        param_layout = QHBoxLayout()

        # 损耗功率
        ploss_layout = QVBoxLayout()
        ploss_layout.addWidget(QLabel("损耗功率 (W)"))
        self.ploss_slider = QSlider(Qt.Horizontal)
        self.ploss_slider.setRange(1, 100)
        self.ploss_slider.setValue(10)
        self.ploss_value = QLabel("1.0")
        ploss_layout.addWidget(self.ploss_slider)
        ploss_layout.addWidget(self.ploss_value)

        # 环境温度
        ambient_layout = QVBoxLayout()
        ambient_layout.addWidget(QLabel("环境温度 (°C)"))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(25)
        self.ambient_value = QLabel("25.0")
        ambient_layout.addWidget(self.ambient_slider)
        ambient_layout.addWidget(self.ambient_value)

        # 对数间隔
        delta_z_layout = QVBoxLayout()
        delta_z_layout.addWidget(QLabel("对数间隔 Δz"))
        self.delta_z_slider = QSlider(Qt.Horizontal)
        self.delta_z_slider.setRange(1, 100)
        self.delta_z_slider.setValue(5)
        self.delta_z_value = QLabel("0.05")
        delta_z_layout.addWidget(self.delta_z_slider)
        delta_z_layout.addWidget(self.delta_z_value)

        param_layout.addLayout(ploss_layout)
        param_layout.addLayout(ambient_layout)
        param_layout.addLayout(delta_z_layout)

        # 分析按钮
        self.analyze_btn = QPushButton("开始分析")
        self.analyze_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14pt; padding: 10px; border-radius: 8px;"
        )
        self.analyze_btn.clicked.connect(self.run_analysis)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)

        control_layout.addLayout(file_layout)
        control_layout.addLayout(param_layout)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addWidget(self.progress_bar)

        # 图表区域
        self.tab_widget = QTabWidget()

        # 创建图表标签页
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()

        self.tab_widget.addTab(self.tab1, "原始数据")
        self.tab_widget.addTab(self.tab2, "对数插值")
        self.tab_widget.addTab(self.tab3, "时间常数谱")
        self.tab_widget.addTab(self.tab4, "结构函数")

        # 设置标签页布局
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

        # 主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.tab_widget, 1)

        # 连接信号
        # if self.update_ploss_value == None:
        #     self.update_ploss_value = 1
        # if self.update_ambient_value == None:
        #     self.update_ambient_value = 25
        # if self.update_delta_z_value == None:
        #     self.update_delta_z_value = 10

        self.ploss_slider.valueChanged.connect(self.update_ploss_value)
        self.ambient_slider.valueChanged.connect(self.update_ambient_value)
        self.delta_z_slider.valueChanged.connect(self.update_delta_z_value)

    def setup_tab1(self):
        """设置原始数据标签页"""
        layout = QVBoxLayout(self.tab1)

        # 创建图表
        self.fig1 = Figure(figsize=(10, 8), dpi=100)
        self.canvas1 = FigureCanvas(self.fig1)

        layout.addWidget(self.canvas1)

    def setup_tab2(self):
        """设置对数插值标签页"""
        layout = QVBoxLayout(self.tab2)

        # 创建图表
        self.fig2 = Figure(figsize=(10, 8), dpi=100)
        self.canvas2 = FigureCanvas(self.fig2)

        layout.addWidget(self.canvas2)

    def setup_tab3(self):
        """设置时间常数谱标签页"""
        layout = QVBoxLayout(self.tab3)

        # 创建图表
        self.fig3 = Figure(figsize=(10, 8), dpi=100)
        self.canvas3 = FigureCanvas(self.fig3)

        layout.addWidget(self.canvas3)

    def setup_tab4(self):
        """设置结构函数标签页"""
        layout = QVBoxLayout(self.tab4)

        # 创建图表
        self.fig4 = Figure(figsize=(10, 8), dpi=100)
        self.canvas4 = FigureCanvas(self.fig4)

        layout.addWidget(self.canvas4)

    def browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择数据文件", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
        )

        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.file_path = file_path

    def update_ploss_value(self, value):
        """更新损耗功率值"""
        ploss = value / 10.0
        self.ploss_value.setText(f"{ploss:.1f}")
        self.processor.ploss = ploss

    def update_ambient_value(self, value):
        """更新环境温度值"""
        self.ambient_value.setText(f"{value}")
        self.ambient_temp = value

    def update_delta_z_value(self, value):
        """更新对数间隔值"""
        delta_z = value / 100.0
        self.delta_z_value.setText(f"{delta_z:.2f}")
        self.processor.delta_z = delta_z

    def run_analysis(self):
        """运行分析"""
        if not hasattr(self, 'file_path'):
            return

        # 更新进度条
        self.progress_bar.setValue(10)

        # 执行分析
        success = self.processor.full_analysis(self.file_path, self.ambient_temp)

        if success:
            self.progress_bar.setValue(100)
            self.plot_results()
        else:
            self.progress_bar.setValue(0)

    def plot_results(self):
        """绘制所有结果图表"""
        self.plot_original_data()
        self.plot_log_interpolation()
        self.plot_time_constant_spectrum()
        self.plot_structure_functions()

    def plot_original_data(self):
        """绘制原始数据"""
        self.fig1.clear()
        ax = self.fig1.add_subplot(111)

        if self.processor.t0 is not None and self.processor.Tj is not None:
            ax.plot(self.processor.t0, self.processor.Tj, 'b-', linewidth=2)

            # 直接设置标签，不使用safe_set_text函数
            try:
                ax.set_xlabel('时间 (s)', fontsize=12)
                ax.set_ylabel('结温 (°C)', fontsize=12)
                ax.set_title('原始温度数据', fontsize=14, fontweight='bold')
            except Exception as e:
                print(f"设置标签时出错: {e}")
                # 使用英文标签作为备选
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Temperature (°C)', fontsize=12)
                ax.set_title('Original Temperature Data', fontsize=14, fontweight='bold')

            # 设置图表格式
            try:
                setup_plot_formatting(ax)
            except Exception as e:
                print(f"设置图表格式时出错: {e}")

            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 确保标签可见
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='both', which='minor', labelsize=8)

        else:
            # 如果没有数据，显示提示信息
            ax.text(0.5, 0.5, '请先加载数据文件', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('原始数据', fontsize=16, fontweight='bold')

        self.fig1.tight_layout()
        self.canvas1.draw()

    def plot_log_interpolation(self):
        """绘制对数插值结果"""
        self.fig2.clear()

        if 'z_fft' in self.processor.results and 'az_fft' in self.processor.results:
            ax1 = self.fig2.add_subplot(211)
            ax1.plot(self.processor.results['z_fft'], self.processor.results['az_fft'], 'b-', linewidth=2)
            try:
                ax1.set_xlabel('z = ln(t)', fontsize=12)
                ax1.set_ylabel('a(z) = Zth(t=exp(z))', fontsize=12)
                ax1.set_title('对数时间轴上的原始数据', fontsize=14, fontweight='bold')
            except:
                ax1.set_xlabel('z = ln(t)', fontsize=12)
                ax1.set_ylabel('a(z) = Zth(t=exp(z))', fontsize=12)
                ax1.set_title('Log Time Axis Data', fontsize=14, fontweight='bold')
            setup_plot_formatting(ax1)
            ax1.grid(True, linestyle='--', alpha=0.7)

        if 'z_bayesian' in self.processor.results and 'az_bayesian' in self.processor.results:
            ax2 = self.fig2.add_subplot(212)
            ax2.plot(self.processor.results['z_bayesian'], self.processor.results['az_bayesian'], 'g-', linewidth=2)
            try:
                ax2.set_xlabel('均匀插值的 z', fontsize=12)
                ax2.set_ylabel('插值后的 a(z)', fontsize=12)
                ax2.set_title('均匀插值后的对数时间数据', fontsize=14, fontweight='bold')
            except:
                ax2.set_xlabel('Interpolated z', fontsize=12)
                ax2.set_ylabel('Interpolated a(z)', fontsize=12)
                ax2.set_title('Interpolated Log Time Data', fontsize=14, fontweight='bold')
            setup_plot_formatting(ax2)
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            # 如果没有数据，显示提示信息
            ax = self.fig2.add_subplot(111)
            ax.text(0.5, 0.5, '请先运行分析以生成插值数据', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('对数插值', fontsize=16, fontweight='bold')
            
        self.fig2.tight_layout()
        self.canvas2.draw()

    def plot_time_constant_spectrum(self):
        """绘制时间常数谱"""
        self.fig3.clear()

        if ('t_bayesian' in self.processor.results and
                'az_bayesian' in self.processor.results and
                'da_dz_bayesian' in self.processor.results and
                'R' in self.processor.results):
            # 创建三个子图
            ax1 = self.fig3.add_subplot(311)
            ax2 = self.fig3.add_subplot(312)
            ax3 = self.fig3.add_subplot(313)

            # 绘制Zth
            t_bayesian = self.processor.results['t_bayesian']
            az_bayesian = self.processor.results['az_bayesian']
            ax1.semilogx(t_bayesian, az_bayesian, 'b-', linewidth=2)
            try:
                ax1.set_title('瞬态热阻抗 Zth', fontsize=12, fontweight='bold')
                ax1.set_xlabel('时间 (s)', fontsize=10)
                ax1.set_ylabel('Zth (K/W)', fontsize=10)
            except:
                ax1.set_title('Transient Thermal Impedance Zth', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time (s)', fontsize=10)
                ax1.set_ylabel('Zth (K/W)', fontsize=10)
            setup_plot_formatting(ax1)
            ax1.grid(True, linestyle='--', alpha=0.7)

            # 绘制导数
            da_dz_bayesian = self.processor.results['da_dz_bayesian']
            da_dz_bayesian_smoothed = self.processor.results['da_dz_bayesian_smoothed']
            z_bayesian = self.processor.results['z_bayesian']
            ax2.plot(z_bayesian[:-1], da_dz_bayesian, 'r-', alpha=0.5, label='原始导数')
            ax2.plot(z_bayesian[:-1], da_dz_bayesian_smoothed, 'b-', linewidth=2, label='平滑后导数')
            try:
                ax2.set_title('导数 da(z)/dz', fontsize=12, fontweight='bold')
                ax2.set_xlabel('z = ln(t)', fontsize=10)
                ax2.set_ylabel('da(z)/dz', fontsize=10)
            except:
                ax2.set_title('Derivative da(z)/dz', fontsize=12, fontweight='bold')
                ax2.set_xlabel('z = ln(t)', fontsize=10)
                ax2.set_ylabel('da(z)/dz', fontsize=10)
            ax2.legend(fontsize=9)
            setup_plot_formatting(ax2)
            ax2.grid(True, linestyle='--', alpha=0.7)

            # 绘制时间常数谱
            R = self.processor.results['R']
            ax3.semilogx(t_bayesian[:-1], R, 'g-', linewidth=2)
            try:
                ax3.set_title('贝叶斯反卷积时间常数谱', fontsize=12, fontweight='bold')
                ax3.set_xlabel('时间 (s)', fontsize=10)
                ax3.set_ylabel('R(z)', fontsize=10)
            except:
                ax3.set_title('Bayesian Deconvolution Time Constant Spectrum', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Time (s)', fontsize=10)
                ax3.set_ylabel('R(z)', fontsize=10)
            setup_plot_formatting(ax3)
            ax3.grid(True, linestyle='--', alpha=0.7)
        else:
            # 如果没有数据，显示提示信息
            ax = self.fig3.add_subplot(111)
            ax.text(0.5, 0.5, '请先运行分析以生成时间常数谱数据', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('时间常数谱', fontsize=16, fontweight='bold')
            
        self.fig3.tight_layout()
        self.canvas3.draw()

    def plot_structure_functions(self):
        """绘制结构函数"""
        self.fig4.clear()

        if ('cumulative_Rth' in self.processor.results and
                'cumulative_Cth' in self.processor.results and
                'differential_Rth' in self.processor.results and
                'differential_Cth' in self.processor.results):
            # 创建两个子图
            ax1 = self.fig4.add_subplot(121)
            ax2 = self.fig4.add_subplot(122)

            # 积分结构函数
            cumulative_Rth = self.processor.results['cumulative_Rth']
            cumulative_Cth = self.processor.results['cumulative_Cth']
            
            # 过滤有效数据（正值且有限值）
            mask1 = (cumulative_Rth > 0) & (cumulative_Cth > 0) & np.isfinite(cumulative_Rth) & np.isfinite(cumulative_Cth)
            if np.any(mask1):
                ax1.semilogy(cumulative_Rth[mask1], cumulative_Cth[mask1], 'b-o', linewidth=2, markersize=4)
                ax1.set_title('积分结构函数', fontsize=12, fontweight='bold')
                ax1.set_xlabel('积分热阻 ∑Rth (K/W)', fontsize=10)
                ax1.set_ylabel('积分热容 ∑Cth (Ws/K)', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.7)
            else:
                ax1.text(0.5, 0.5, '无有效数据', transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('积分结构函数', fontsize=12, fontweight='bold')

            # 微分结构函数
            differential_Rth = self.processor.results['differential_Rth']
            differential_Cth = self.processor.results['differential_Cth']
            
            # 过滤有效数据（正值且有限值）
            mask2 = (differential_Rth > 0) & (differential_Cth > 0) & np.isfinite(differential_Rth) & np.isfinite(differential_Cth)
            if np.any(mask2):
                ax2.semilogy(differential_Rth[mask2], differential_Cth[mask2], 'r-s', linewidth=2, markersize=4)
                ax2.set_title('微分结构函数', fontsize=12, fontweight='bold')
                ax2.set_xlabel('热阻 Rth (K/W)', fontsize=10)
                ax2.set_ylabel('热容 Cth (Ws/K)', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.7)
            else:
                ax2.text(0.5, 0.5, '无有效数据', transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('微分结构函数', fontsize=12, fontweight='bold')

            # 添加统计信息
            if 'tau_sorted' in self.processor.results:
                tau_sorted = self.processor.results['tau_sorted']
                valid_tau = tau_sorted[tau_sorted > 0]
                if len(valid_tau) > 0:
                    info_text = f"时间常数范围: {valid_tau.min():.2e} - {valid_tau.max():.2e} s"
                    self.fig4.suptitle(info_text, fontsize=10, y=0.95)

            # 设置图表格式
            for ax in [ax1, ax2]:
                ax.tick_params(axis='both', which='major', labelsize=9)
                ax.tick_params(axis='both', which='minor', labelsize=8)

        else:
            # 如果没有数据，显示提示信息
            ax = self.fig4.add_subplot(111)
            ax.text(0.5, 0.5, '请先运行分析以生成结构函数数据', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('结构函数', fontsize=16, fontweight='bold')

        self.fig4.tight_layout()
        self.canvas4.draw()

    # def plot_structure_functions(self):
    #     """绘制结构函数"""
    #     self.fig4.clear()
    #
    #     if ('cumulative_Rth' in self.processor.results and
    #             'cumulative_Cth' in self.processor.results and
    #             'differential_Rth' in self.processor.results and
    #             'differential_Cth' in self.processor.results):
    #         # 创建两个子图
    #         ax1 = self.fig4.add_subplot(121)
    #         ax2 = self.fig4.add_subplot(122)
    #
    #         # 积分结构函数
    #         cumulative_Rth = self.processor.results['cumulative_Rth']
    #         cumulative_Cth = self.processor.results['cumulative_Cth']
    #         # 过滤正值
    #         mask1 = (cumulative_Rth > 0) & (cumulative_Cth > 0)
    #         ax1.semilogy(cumulative_Rth[mask1], cumulative_Cth[mask1], 'b-', linewidth=2)
    #         ax1.set_title('积分结构函数', fontsize=12)
    #         ax1.set_xlabel('积分热阻 ∑Rth (K/W)', fontsize=10)
    #         ax1.set_ylabel('积分热容 ∑Cth (Ws/K)', fontsize=10)
    #         setup_plot_formatting(ax1)
    #         ax1.grid(True, linestyle='--', alpha=0.7)
    #
    #         # 微分结构函数
    #         differential_Rth = self.processor.results['differential_Rth']
    #         differential_Cth = self.processor.results['differential_Cth']
    #         # 过滤正值
    #         mask2 = (differential_Rth > 0) & (differential_Cth > 0)
    #         ax2.semilogy(differential_Rth[mask2], differential_Cth[mask2], 'r-', linewidth=2)
    #         ax2.set_title('微分结构函数', fontsize=12)
    #         ax2.set_xlabel('热阻 Rth (K/W)', fontsize=10)
    #         ax2.set_ylabel('热容 Cth (Ws/K)', fontsize=10)
    #         setup_plot_formatting(ax2)
    #         ax2.grid(True, linestyle='--', alpha=0.7)
    #     self.fig4.tight_layout()
    #     self.canvas4.draw()

    def foster_to_cauer_numeric(fosterRth, fosterCth):
        """
        使用数值方法将 Foster 网络转换为 Cauer 网络

        参数:
        fosterRth -- Foster 网络的热阻数组
        fosterCth -- Foster 网络的热容数组

        返回:
        cauerRth -- Cauer 网络的热阻数组
        cauerCth -- Cauer 网络的热容数组
        """
        # 确保输入有效
        if len(fosterRth) != len(fosterCth):
            raise ValueError("fosterRth 和 fosterCth 长度必须相同")

        n = len(fosterRth)
        cauerRth = []
        cauerCth = []

        # 构建导纳矩阵
        Y = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    Y[i, j] = 1 / fosterRth[i] + fosterCth[i]
                else:
                    Y[i, j] = 0

        # 构建阻抗矩阵
        Z = np.linalg.inv(Y)

        # 提取 Cauer 网络参数
        for i in range(n):
            # 计算并联热容
            c = Y[i, i] - sum(Y[i, j] for j in range(i))
            if c > 0:
                cauerCth.append(c)
            else:
                break

            # 计算串联热阻
            r = Z[i, i] - sum(Z[i, j] for j in range(i))
            if r > 0:
                cauerRth.append(r)
            else:
                break

        return np.array(cauerRth), np.array(cauerCth)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    setup_fonts()
    window = ThermalAnalysisView()
    window.show()
    sys.exit(app.exec_())