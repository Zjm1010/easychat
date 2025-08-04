#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
热分析系统UI窗口模块
包含主窗口界面和所有UI相关的功能
"""

import os
# 导入处理器和工具函数
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGroupBox, QFileDialog,
                             QProgressBar, QTabWidget)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(os.path.join(os.path.dirname(__file__), ''))
from decov_function import ThermalAnalysisProcessor, setup_fonts, setup_plot_formatting


class ThermalAnalysisView(QMainWindow):
    """热分析系统主窗口类"""
    
    def __init__(self):
        super().__init__()
        # 初始化属性
        self.ambient_temp = 25.0  # 默认环境温度
        self.ploss = 1.0  # 默认损耗功率
        self.delta_z = 0.05  # 默认对数间隔
        self.num_iterations = 500  # 默认迭代次数
        self.discrete_order = 45  # 默认离散阶数
        self.precision = 'float64'  # 默认计算精度
        self.results = {}  # 初始化结果字典

        # 创建处理器实例
        self.processor = ThermalAnalysisProcessor(precision=self.precision)

        self.init_ui()
        self.setWindowTitle("Bayesian Deconvolution Thermal Analysis System")
        self.setGeometry(100, 100, 1200, 800)

    def init_ui(self):
        """初始化用户界面"""
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title_layout = QVBoxLayout()
        title_label = QLabel("Bayesian Deconvolution Thermal Analysis System")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; margin: 10px 0;")

        subtitle_label = QLabel("Thermal Resistance Extraction and Analysis Based on Structure Function Method")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 14pt; color: #666; margin-bottom: 20px;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        main_layout.addLayout(title_layout)

        # 控制面板
        control_group = QGroupBox("Control Panel")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(15, 15, 15, 15)
        control_group.setStyleSheet(
            "QGroupBox {border: 1px solid #ddd; border-radius: 8px; padding: 10px;}"
            "QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 5px;}"
        )

        # 文件选择
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("border: 1px solid #ddd; padding: 5px; border-radius: 4px;")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setStyleSheet("padding: 5px 15px;")
        self.browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(QLabel("Data File:"))
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.browse_btn)

        # 参数设置
        param_layout = QHBoxLayout()

        # 损耗功率
        ploss_layout = QVBoxLayout()
        ploss_layout.addWidget(QLabel("Power Loss (W)"))
        self.ploss_slider = QSlider(Qt.Horizontal)
        self.ploss_slider.setRange(1, 100)
        self.ploss_slider.setValue(10)
        self.ploss_value = QLabel("1.0")
        ploss_layout.addWidget(self.ploss_slider)
        ploss_layout.addWidget(self.ploss_value)

        # 环境温度
        ambient_layout = QVBoxLayout()
        ambient_layout.addWidget(QLabel("Ambient Temperature (°C)"))
        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setRange(0, 100)
        self.ambient_slider.setValue(25)
        self.ambient_value = QLabel("25.0")
        ambient_layout.addWidget(self.ambient_slider)
        ambient_layout.addWidget(self.ambient_value)

        # 对数间隔
        delta_z_layout = QVBoxLayout()
        delta_z_layout.addWidget(QLabel("Log Interval Δz"))
        self.delta_z_slider = QSlider(Qt.Horizontal)
        self.delta_z_slider.setRange(1, 100)
        self.delta_z_slider.setValue(5)
        self.delta_z_value = QLabel("0.05")
        delta_z_layout.addWidget(self.delta_z_slider)
        delta_z_layout.addWidget(self.delta_z_value)

        # 计算精度
        precision_layout = QVBoxLayout()
        precision_layout.addWidget(QLabel("Calculation Precision"))
        from PyQt5.QtWidgets import QComboBox
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(['float64', 'float32'])
        self.precision_combo.setCurrentText(self.precision)
        self.precision_combo.currentTextChanged.connect(self.update_precision)
        precision_layout.addWidget(self.precision_combo)

        param_layout.addLayout(ploss_layout)
        param_layout.addLayout(ambient_layout)
        param_layout.addLayout(delta_z_layout)
        param_layout.addLayout(precision_layout)

        # 分析按钮
        self.analyze_btn = QPushButton("Start Analysis")
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

        self.tab_widget.addTab(self.tab1, "Original Data")
        self.tab_widget.addTab(self.tab2, "Log Interpolation")
        self.tab_widget.addTab(self.tab3, "Time Constant Spectrum")
        self.tab_widget.addTab(self.tab4, "Structure Function")

        # 设置标签页布局
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()

        # 主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.tab_widget, 1)

        # 连接信号
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
            self, "Select Data File", "", "Excel Files (*.xlsx *.xls);;All Files (*)"
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

    def update_precision(self, precision):
        """更新计算精度"""
        self.precision = precision
        # 重新创建处理器实例以应用新的精度设置
        self.processor = ThermalAnalysisProcessor(precision=self.precision)
        print(f"计算精度已更新为: {precision}")

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
                ax.set_xlabel('Time (s)', fontsize=12)
                ax.set_ylabel('Junction Temperature (°C)', fontsize=12)
                ax.set_title('Original Temperature Data', fontsize=14, fontweight='bold')
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
            ax.text(0.5, 0.5, 'Please load data file first',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Original Data', fontsize=16, fontweight='bold')

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
                ax1.set_title('Original Data on Log Time Axis', fontsize=14, fontweight='bold')
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
                ax2.set_xlabel('Uniformly Interpolated z', fontsize=12)
                ax2.set_ylabel('Interpolated a(z)', fontsize=12)
                ax2.set_title('Uniformly Interpolated Log Time Data', fontsize=14, fontweight='bold')
            except:
                ax2.set_xlabel('Interpolated z', fontsize=12)
                ax2.set_ylabel('Interpolated a(z)', fontsize=12)
                ax2.set_title('Interpolated Log Time Data', fontsize=14, fontweight='bold')
            setup_plot_formatting(ax2)
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            # 如果没有数据，显示提示信息
            ax = self.fig2.add_subplot(111)
            ax.text(0.5, 0.5, 'Please run analysis to generate interpolation data',
                    transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Log Interpolation', fontsize=16, fontweight='bold')

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
                ax1.set_title('Transient thermal impedance Zth', fontsize=12, fontweight='bold')
                ax1.set_xlabel('Time (s)', fontsize=10)
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
            ax2.plot(z_bayesian[:-1], da_dz_bayesian, 'r-', alpha=0.5, label='Original derivative')
            ax2.plot(z_bayesian[:-1], da_dz_bayesian_smoothed, 'b-', linewidth=2, label='Derivative after smoothing')
            try:
                ax2.set_title('Derivative da(z)/dz', fontsize=12, fontweight='bold')
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
                ax3.set_title('Bayesian Deconvolution Time Constant Spectrum', fontsize=12, fontweight='bold')
                ax3.set_xlabel('Time (s)', fontsize=10)
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
            ax.text(0.5, 0.5, 'Run time constant spectrum first',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Time constant spectrum', fontsize=16, fontweight='bold')
            
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
                ax1.set_title('Integral structure function', fontsize=12, fontweight='bold')
                ax1.set_xlabel('thermal resistance ∑Rth (K/W)', fontsize=10)
                ax1.set_ylabel('thermal capacity ∑Cth (Ws/K)', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.7)
                # 设置纵坐标最大值为10^65
                ax1.set_ylim(bottom=ax1.get_ylim()[0], top=1e3)
            else:
                ax1.text(0.5, 0.5, 'no valid data', transform=ax1.transAxes, ha='center', va='center')
                ax1.set_title('Integral structure function', fontsize=12, fontweight='bold')

            # 微分结构函数
            differential_Rth = self.processor.results['differential_Rth']
            differential_Cth = self.processor.results['differential_Cth']
            
            # 确保转换为numpy数组
            try:
                differential_Rth = np.array(differential_Rth, dtype=float)
                differential_Cth = np.array(differential_Cth, dtype=float)
            except (ValueError, TypeError) as e:
                print(f"警告: 微分结构函数数据转换失败: {e}")
                # 如果微分数据转换失败，只显示积分结构函数
                differential_Rth = np.array([])
                differential_Cth = np.array([])
            
            # 过滤有效数据（正值且有限值）
            mask2 = (differential_Rth > 0) & (differential_Cth > 0) & np.isfinite(differential_Rth) & np.isfinite(differential_Cth)
            if np.any(mask2):
                ax2.semilogy(differential_Rth[mask2], differential_Cth[mask2], 'r-s', linewidth=2, markersize=4)
                ax2.set_title('Differential structure function', fontsize=12, fontweight='bold')
                ax2.set_xlabel('thermal resistance Rth (K/W)', fontsize=10)
                ax2.set_ylabel('thermal capacity Cth (Ws/K)', fontsize=10)
                ax2.grid(True, linestyle='--', alpha=0.7)
                # 设置纵坐标最大值为10^65
                ax2.set_ylim(bottom=ax2.get_ylim()[0], top=1e12)
            else:
                ax2.text(0.5, 0.5, 'no valid data', transform=ax2.transAxes, ha='center', va='center')
                ax2.set_title('Differential structure function', fontsize=12, fontweight='bold')

            # 添加统计信息
            info_text = ""
            if len(cumulative_Rth) > 0:
                info_text += f"data point num: {len(cumulative_Rth)}"
            if len(differential_Rth) > 0:
                if info_text:
                    info_text += " | "
                info_text += f"data point num: {len(differential_Rth)}"
            
            if info_text:
                self.fig4.suptitle(info_text, fontsize=10, y=0.95)

            # 设置图表格式
            for ax in [ax1, ax2]:
                ax.tick_params(axis='both', which='major', labelsize=9)
                ax.tick_params(axis='both', which='minor', labelsize=8)

        else:
            # 如果没有数据，显示提示信息
            ax = self.fig4.add_subplot(111)
            ax.text(0.5, 0.5, 'Run structure function first',
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('structure function', fontsize=16, fontweight='bold')

        self.fig4.tight_layout()
        self.canvas4.draw()


def create_main_window():
    """创建并返回主窗口实例"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    setup_fonts()
    window = ThermalAnalysisView()
    return window


if __name__ == '__main__':
    window = create_main_window()
    window.show()
    sys.exit(QApplication.instance().exec_())
