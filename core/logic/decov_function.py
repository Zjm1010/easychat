import os

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from matplotlib import ticker
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


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
        """时间常数谱离散化 - 改进版本"""
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

        # 改进：使用更智能的区间划分方法
        # 基于R值的分布进行自适应区间划分
        R_cumsum = np.cumsum(R_valid)
        R_total = R_cumsum[-1]
        
        # 确保每个区间包含足够的R值
        min_R_per_interval = R_total / (self.discrete_order * 2)  # 至少包含总R值的1/(2*order)
        
        # 创建区间边界
        interval_boundaries = []
        current_sum = 0
        target_sum = R_total / self.discrete_order
        
        for i, r_val in enumerate(R_valid):
            current_sum += r_val
            if current_sum >= target_sum and len(interval_boundaries) < self.discrete_order - 1:
                interval_boundaries.append(i)
                current_sum = 0
        
        # 确保有足够的区间
        if len(interval_boundaries) < self.discrete_order - 1:
            # 如果区间不够，使用均匀划分
            step = len(R_valid) // self.discrete_order
            interval_boundaries = [i * step for i in range(1, self.discrete_order)]
        
        # 添加起始和结束边界
        interval_boundaries = [0] + interval_boundaries + [len(R_valid)]
        
        # Foster网络参数计算
        fosterRth = []
        fosterCth = []
        tau_Foster = []

        for i in range(len(interval_boundaries) - 1):
            start_idx = interval_boundaries[i]
            end_idx = interval_boundaries[i + 1]
            
            if end_idx > start_idx:
                # 计算该区间内的总热阻
                interval_R = np.sum(R_valid[start_idx:end_idx]) * self.delta_z
                
                if interval_R > 0:
                    # 计算该区间的平均时间常数
                    z_interval = z_valid[start_idx:end_idx]
                    tau_interval = np.exp(z_interval)
                    
                    # 使用加权平均计算时间常数
                    weights = R_valid[start_idx:end_idx]
                    tau_weighted = np.average(tau_interval, weights=weights)
                    
                    # 计算对应的热容
                    interval_C = tau_weighted / interval_R
                    
                    # 验证热容的合理性
                    if interval_C > 0 and np.isfinite(interval_C):
                        fosterRth.append(interval_R)
                        fosterCth.append(interval_C)
                        tau_Foster.append(interval_R * interval_C)

        # 转换为numpy数组
        fosterRth = np.array(fosterRth)
        fosterCth = np.array(fosterCth)
        tau_Foster = np.array(tau_Foster)

        # 过滤掉零值和负值
        valid_foster_mask = (fosterRth > 0) & (fosterCth > 0) & np.isfinite(fosterRth) & np.isfinite(fosterCth)
        if not np.any(valid_foster_mask):
            print("警告: 没有生成有效的Foster网络参数")
            return False

        self.results['fosterRth'] = fosterRth[valid_foster_mask]
        self.results['fosterCth'] = fosterCth[valid_foster_mask]
        self.results['tau_Foster'] = tau_Foster[valid_foster_mask]

        print(f"改进的Foster网络参数计算完成: {np.sum(valid_foster_mask)} 个有效参数")
        print(f"热阻范围: {fosterRth[valid_foster_mask].min():.2e} - {fosterRth[valid_foster_mask].max():.2e} K/W")
        print(f"热容范围: {fosterCth[valid_foster_mask].min():.2e} - {fosterCth[valid_foster_mask].max():.2e} Ws/K")
        return True



    def foster_to_cauer(self):
        foster_rth = self.results['fosterRth']
        foster_cth = self.results['fosterCth']
        s = sp.symbols('s')
        # 计算复阻抗 Z(s)
        Zs = 0
        for i in range(len(foster_rth)):
            Zs += foster_rth[i] / (1 + foster_rth[i] * foster_cth[i] * s)

        # 计算复导纳 Y(s)
        Ys = 1 / Zs
        Ys = sp.simplify(Ys)

        # 初始化Cauer网络参数列表
        cauerRth = []
        cauerCth = []

        n = len(foster_rth)
        # 迭代提取Cauer参数
        for i in range(n):
            # 提取并联电容（通过计算s->oo时 Y(s)/s 的极限）
            C_val = sp.limit(Ys / s, s, sp.oo)
            # 如果无法提取电容则终止迭代
            if sp.simplify(C_val).is_zero or C_val == sp.oo:
                break
            cauerCth.append(C_val)

            # 更新导纳：移除并联电容项
            Ys = sp.expand(Ys - C_val * s)
            Ys = sp.simplify(Ys)

            # 如果导纳为0则终止迭代
            if Ys == 0:
                break

            # 计算新阻抗 Z(s) = 1/Y(s)
            Zs = 1 / Ys
            Zs = sp.simplify(Zs)

            # 提取串联电阻（通过计算s->0时 Z(s) 的极限）
            R_val = sp.limit(Zs, s, 0)
            # 如果无法提取电阻则终止迭代
            if sp.simplify(R_val).is_zero or R_val == sp.oo:
                break
            cauerRth.append(R_val)

            # 更新阻抗：移除串联电阻项
            Zs = sp.expand(Zs - R_val)
            Zs = sp.simplify(Zs)

            # 如果阻抗为0则终止迭代
            if Zs == 0:
                break
            # 更新导纳为剩余阻抗的倒数
            Ys = 1 / Zs
            Ys = sp.simplify(Ys)
        self.results['cauerRth'] = cauerRth
        self.results['cauerCth'] = cauerCth
        return True

    def calculate_structure_functions(self):
        """计算结构函数 - 基于Cauer网络参数
        
        注意：结构函数应该基于Cauer网络参数而不是Foster网络参数，因为：
        1. Cauer网络是梯形网络，直接对应物理结构
        2. Cauer网络的热阻和热容按物理层次排列，适合计算结构函数
        3. 积分结构函数表示从热源到环境的总热阻和热容
        4. 微分结构函数表示每个物理层的热阻和热容分布
        
        计算逻辑基于Matlab代码：
        - 积分结构函数：cumsum(cauerRth) 和 cumsum(cauerCth)
        - 微分结构函数：diff(cumulative_Cth) / diff(cumulative_Rth)
        - 自动去除热容过大的异常层
        """
        if ('cauerRth' not in self.results or
                'cauerCth' not in self.results):
            print("警告: 没有Cauer网络参数，请先执行Foster到Cauer转换")
            return False

        cauerRth = self.results['cauerRth']
        cauerCth = self.results['cauerCth']

        # 过滤掉零值和负值
        valid_mask = (cauerRth > 0) & (cauerCth > 0) & np.isfinite(cauerRth) & np.isfinite(cauerCth)
        if not np.any(valid_mask):
            print("警告: 没有有效的Cauer网络参数用于结构函数计算")
            return False

        cauerRth_valid = cauerRth[valid_mask]
        cauerCth_valid = cauerCth[valid_mask]

        # Cauer网络的结构函数计算 - 基于Matlab代码逻辑
        # 积分结构函数计算
        cumulative_Rth = np.cumsum(cauerRth_valid)
        cumulative_Cth = np.cumsum(cauerCth_valid)

        # 计算微分结构函数
        # 注意：避免除零错误
        if len(cumulative_Rth) > 1:
            # 长度正确的数组
            differential_Cth = np.zeros(len(cumulative_Rth) - 1)
            differential_Rth = np.zeros(len(cumulative_Rth) - 1)

            # 计算差分
            diff_cumulative_Rth = np.diff(cumulative_Rth)
            diff_cumulative_Cth = np.diff(cumulative_Cth)

            # 处理有效差分
            valid_diff_mask = diff_cumulative_Rth > 0
            if np.any(valid_diff_mask):
                differential_Cth[valid_diff_mask] = diff_cumulative_Cth[valid_diff_mask] / diff_cumulative_Rth[
                    valid_diff_mask]
                differential_Rth[valid_diff_mask] = cumulative_Rth[:-1][valid_diff_mask]

            # 过滤无效值
            valid_mask = (differential_Rth > 0) & (differential_Cth > 0) & np.isfinite(differential_Rth) & np.isfinite(
                differential_Cth)
            differential_Rth = differential_Rth[valid_mask]
            differential_Cth = differential_Cth[valid_mask]
        else:
            # 如果只有一个数据点，则没有微分值
            differential_Rth = np.array([])
            differential_Cth = np.array([])
        
        # 去掉最后几个热容特别大的层（如果存在）
        if len(cumulative_Cth) > 4:
            # 计算热容的阈值（例如：超过平均值的10倍）
            mean_Cth = np.mean(cumulative_Cth)
            threshold = mean_Cth * 10
            
            # 找到需要保留的索引
            valid_indices = cumulative_Cth <= threshold
            
            if np.sum(valid_indices) < len(cumulative_Cth):
                print(f"去掉了 {len(cumulative_Cth) - np.sum(valid_indices)} 个热容过大的层")
                cumulative_Rth = cumulative_Rth[valid_indices]
                cumulative_Cth = cumulative_Cth[valid_indices]
                
                # 同时更新微分结构函数
                if len(differential_Rth) > 0:
                    # 确保微分结构函数的长度与积分结构函数匹配
                    max_valid_index = np.max(np.where(valid_indices)[0])
                    if max_valid_index < len(differential_Rth):
                        differential_Rth = differential_Rth[:max_valid_index]
                        differential_Cth = differential_Cth[:max_valid_index]

        # 保存结果
        self.results['cumulative_Rth'] = cumulative_Rth
        self.results['cumulative_Cth'] = cumulative_Cth
        self.results['differential_Rth'] = differential_Rth
        self.results['differential_Cth'] = differential_Cth

        return True

    def calculate_structure_functions_alternative(self):
        """备选的结构函数计算方法 - 基于Cauer网络参数
        
        这是备选的结构函数计算方法，使用相同的Cauer网络参数但可能有不同的处理逻辑。
        计算逻辑与主要方法相同，基于Matlab代码实现。
        """
        if ('cauerRth' not in self.results or
                'cauerCth' not in self.results):
            print("警告: 没有Cauer网络参数，请先执行Foster到Cauer转换")
            return False

        cauerRth = self.results['cauerRth']
        cauerCth = self.results['cauerCth']

        # 过滤掉零值和负值
        valid_mask = (cauerRth > 0) & (cauerCth > 0) & np.isfinite(cauerRth) & np.isfinite(cauerCth)
        if not np.any(valid_mask):
            print("警告: 没有有效的Cauer网络参数用于结构函数计算")
            return False

        cauerRth_valid = cauerRth[valid_mask]
        cauerCth_valid = cauerCth[valid_mask]
        # Cauer网络的结构函数计算 - 基于Matlab代码逻辑
        # 积分结构函数计算
        cumulative_Rth = np.cumsum(cauerRth_valid)
        cumulative_Cth = np.cumsum(cauerCth_valid)
        
        # 微分结构函数计算 - 按照Matlab代码逻辑
        # 计算相邻积分值的差值
        diff_cumulative_Rth = np.diff(cumulative_Rth)
        diff_cumulative_Cth = np.diff(cumulative_Cth)
        
        # 计算微分结构函数
        # 注意：避免除零错误
        valid_diff_mask = diff_cumulative_Rth > 0
        if np.any(valid_diff_mask):
            differential_Cth = np.zeros_like(cauerCth_valid)
            differential_Rth = np.zeros_like(cauerRth_valid)
            
            # 对于有效差值，计算微分热容
            differential_Cth[valid_diff_mask] = diff_cumulative_Cth[valid_diff_mask] / diff_cumulative_Rth[valid_diff_mask]
            differential_Rth[valid_diff_mask] = cumulative_Rth[:-1][valid_diff_mask]
            
            # 过滤掉无效值
            valid_mask = (differential_Rth > 0) & (differential_Cth > 0) & np.isfinite(differential_Rth) & np.isfinite(differential_Cth)
            differential_Rth = differential_Rth[valid_mask]
            differential_Cth = differential_Cth[valid_mask]
        else:
            # 如果没有有效差值，使用原始Cauer参数
            differential_Rth = cauerRth_valid
            differential_Cth = cauerCth_valid
        
        # 去掉最后几个热容特别大的层（如果存在）
        if len(cumulative_Cth) > 4:
            # 计算热容的阈值（例如：超过平均值的10倍）
            mean_Cth = np.mean(cumulative_Cth)
            threshold = mean_Cth * 10
            
            # 找到需要保留的索引
            valid_indices = cumulative_Cth <= threshold
            
            if np.sum(valid_indices) < len(cumulative_Cth):
                print(f"去掉了 {len(cumulative_Cth) - np.sum(valid_indices)} 个热容过大的层")
                cumulative_Rth = cumulative_Rth[valid_indices]
                cumulative_Cth = cumulative_Cth[valid_indices]
                
                # 同时更新微分结构函数
                if len(differential_Rth) > 0:
                    # 确保微分结构函数的长度与积分结构函数匹配
                    max_valid_index = np.max(np.where(valid_indices)[0])
                    if max_valid_index < len(differential_Rth):
                        differential_Rth = differential_Rth[:max_valid_index]
                        differential_Cth = differential_Cth[:max_valid_index]

        # 保存结果
        self.results['cumulative_Rth'] = cumulative_Rth
        self.results['cumulative_Cth'] = cumulative_Cth
        self.results['differential_Rth'] = differential_Rth
        self.results['differential_Cth'] = differential_Cth

        print(f"基于Cauer网络的备选结构函数计算完成: {len(cumulative_Rth)} 个数据点")
        print(f"热阻范围: {differential_Rth.min():.2e} - {differential_Rth.max():.2e} K/W")
        print(f"积分热容范围: {cumulative_Cth.min():.2e} - {cumulative_Cth.max():.2e} Ws/K")
        print(f"微分热容范围: {differential_Cth.min():.2e} - {differential_Cth.max():.2e} Ws/K")
        
        # 验证积分结构函数的阶梯状特性
        if len(cumulative_Cth) > 1:
            steps = np.diff(cumulative_Cth)
            print(f"积分热容阶梯数: {len(steps)}")
            print(f"阶梯高度范围: {steps.min():.2e} - {steps.max():.2e} Ws/K")
            print(f"平均阶梯高度: {steps.mean():.2e} Ws/K")
        
        return True

    def debug_structure_functions(self):
        """调试结构函数计算过程"""
        print("\n=== 结构函数计算调试信息 ===")
        
        if 'cauerRth' in self.results and 'cauerCth' in self.results:
            cauerRth = self.results['cauerRth']
            cauerCth = self.results['cauerCth']
            
            print(f"Cauer网络参数数量: {len(cauerRth)}")
            print(f"热阻范围: {cauerRth.min():.2e} - {cauerRth.max():.2e} K/W")
            print(f"热容范围: {cauerCth.min():.2e} - {cauerCth.max():.2e} Ws/K")
            
            # 检查时间常数
            tau = cauerRth * cauerCth
            print(f"时间常数范围: {tau.min():.2e} - {tau.max():.2e} s")
            
            # 检查有效参数
            valid_mask = (cauerRth > 0) & (cauerCth > 0) & np.isfinite(cauerRth) & np.isfinite(cauerCth)
            print(f"有效参数数量: {np.sum(valid_mask)}")
            
            if np.any(valid_mask):
                print("前5个有效参数:")
                for i in range(min(5, np.sum(valid_mask))):
                    idx = np.where(valid_mask)[0][i]
                    print(f"  R{i+1}={cauerRth[idx]:.2e} K/W, C{i+1}={cauerCth[idx]:.2e} Ws/K, τ{i+1}={tau[idx]:.2e} s")
        else:
            print("未找到Cauer网络参数")
            
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

    def analyze_thermal_capacity_distribution(self):
        """分析热容值分布，帮助诊断积分结构函数问题"""
        print("\n=== 热容值分布分析 ===")
        
        if 'cauerRth' in self.results and 'cauerCth' in self.results:
            cauerRth = self.results['cauerRth']
            cauerCth = self.results['cauerCth']
            
            # 分析Cauer网络热容值的分布
            print(f"Cauer网络热容值统计:")
            print(f"  最小值: {cauerCth.min():.2e} Ws/K")
            print(f"  最大值: {cauerCth.max():.2e} Ws/K")
            print(f"  平均值: {cauerCth.mean():.2e} Ws/K")
            print(f"  中位数: {np.median(cauerCth):.2e} Ws/K")
            print(f"  标准差: {cauerCth.std():.2e} Ws/K")
            
            # # 检查热容值的分布是否均匀
            # sorted_Cth = np.sort(cauerCth)
            # differences = np.diff(sorted_Cth)
            # print(f"  热容值相邻差值范围: {differences.min():.2e} - {differences.max():.2e} Ws/K")
            # print(f"  热容值相邻差值平均值: {differences.mean():.2e} Ws/K")
            #
            # # 检查是否有重复或相近的热容值
            # unique_Cth, counts = np.unique(cauerCth, return_counts=True)
            # print(f"  唯一热容值数量: {len(unique_Cth)} / {len(cauerCth)}")
            # if len(unique_Cth) < len(cauerCth):
            #     print(f"  存在重复热容值，重复次数最多的值出现 {counts.max()} 次")
            #
            # # 分析时间常数与热容的关系
            # tau = cauerRth * cauerCth
            # print(f"\n时间常数与热容关系:")
            # print(f"  时间常数范围: {tau.min():.2e} - {tau.max():.2e} s")
            #
            # # 检查是否存在异常的时间常数
            # tau_ratio = tau.max() / tau.min()
            # print(f"  时间常数比值 (max/min): {tau_ratio:.2e}")
            #
            # if tau_ratio > 1e6:
            #     print("  警告: 时间常数范围过大，可能导致热容计算不准确")
        
        if 'fosterRth' in self.results and 'fosterCth' in self.results:
            fosterRth = self.results['fosterRth']
            fosterCth = self.results['fosterCth']
            
            # 分析Foster网络热容值的分布
            print(f"\nFoster网络热容值统计:")
            print(f"  最小值: {fosterCth.min():.2e} Ws/K")
            print(f"  最大值: {fosterCth.max():.2e} Ws/K")
            print(f"  平均值: {fosterCth.mean():.2e} Ws/K")
            print(f"  中位数: {np.median(fosterCth):.2e} Ws/K")
            print(f"  标准差: {fosterCth.std():.2e} Ws/K")
            
            # 检查热容值的分布是否均匀
            sorted_Cth = np.sort(fosterCth)
            differences = np.diff(sorted_Cth)
            print(f"  热容值相邻差值范围: {differences.min():.2e} - {differences.max():.2e} Ws/K")
            print(f"  热容值相邻差值平均值: {differences.mean():.2e} Ws/K")
            
            # 检查是否有重复或相近的热容值
            unique_Cth, counts = np.unique(fosterCth, return_counts=True)
            print(f"  唯一热容值数量: {len(unique_Cth)} / {len(fosterCth)}")
            if len(unique_Cth) < len(fosterCth):
                print(f"  存在重复热容值，重复次数最多的值出现 {counts.max()} 次")
            
            # 分析时间常数与热容的关系
            tau = fosterRth * fosterCth
            print(f"\n时间常数与热容关系:")
            print(f"  时间常数范围: {tau.min():.2e} - {tau.max():.2e} s")
            
            # 检查是否存在异常的时间常数
            tau_ratio = tau.max() / tau.min()
            print(f"  时间常数比值 (max/min): {tau_ratio:.2e}")
            
            if tau_ratio > 1e6:
                print("  警告: 时间常数范围过大，可能导致热容计算不准确")
            
        if 'cumulative_Cth' in self.results:
            cumulative_Cth = self.results['cumulative_Cth']
            print(f"\n积分结构函数热容分析:")
            print(f"  积分热容变化范围: {cumulative_Cth.max() - cumulative_Cth.min():.2e} Ws/K")
            
            # 检查积分热容的单调性
            if np.all(np.diff(cumulative_Cth) >= 0):
                print("  积分热容单调递增 ✓")
            else:
                print("  警告: 积分热容不是单调递增的")
            
            # 分析积分结构函数的阶梯状特性
            if len(cumulative_Cth) > 1:
                steps = np.diff(cumulative_Cth)
                print(f"  积分热容阶梯数: {len(steps)}")
                print(f"  阶梯高度范围: {steps.min():.2e} - {steps.max():.2e} Ws/K")
                print(f"  平均阶梯高度: {steps.mean():.2e} Ws/K")
                print(f"  阶梯高度标准差: {steps.std():.2e} Ws/K")
                
                # 检查阶梯的均匀性
                step_ratio = steps.max() / steps.min() if steps.min() > 0 else float('inf')
                print(f"  阶梯高度比值 (max/min): {step_ratio:.2e}")
                
                if step_ratio < 10:
                    print("  阶梯高度分布相对均匀 ✓")
                elif step_ratio < 100:
                    print("  阶梯高度分布中等不均匀")
                else:
                    print("  警告: 阶梯高度分布极不均匀")
                
                # 检查零阶梯（平台）
                zero_steps = np.sum(steps == 0)
                small_steps = np.sum(steps < 1e-10)
                print(f"  零阶梯数: {zero_steps}")
                print(f"  极小阶梯数 (<1e-10): {small_steps}")
                
                if zero_steps == 0 and small_steps < len(steps) * 0.1:
                    print("  积分结构函数阶梯状特性正常 ✓")
                else:
                    print("  警告: 积分结构函数可能存在异常的平台或极小阶梯")
            
        print("=== 热容值分布分析结束 ===\n")

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

        # 尝试连分式展开法进行Foster到Cauer转换
        if not self.foster_to_cauer():
            return False

        # 基于Cauer网络计算结构函数
        if not self.calculate_structure_functions():
            # 如果主要方法失败，尝试备选方法
            if not self.calculate_structure_functions_alternative():
                print("警告: 结构函数计算失败")
                return False

        # 添加调试信息
        # self.debug_structure_functions()
        # self.analyze_thermal_capacity_distribution()
        # self.analyze_cumulative_thermal_capacity_issue()

        return True
