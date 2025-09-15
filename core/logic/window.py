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

# Qt compatibility (support PyQt5 and PyQt6)
QT_VERSION = None
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSlider, QLabel, QPushButton, QGroupBox, QFileDialog,
        QProgressBar, QTabWidget, QMessageBox, QTableWidgetItem,
        QComboBox, QLineEdit, QTextEdit, QSplitter, QTableWidget
    )
    from PyQt5.QtGui import QColor
    QT_VERSION = 5
    print("Using PyQt5")
except ImportError:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QSlider, QLabel, QPushButton, QGroupBox, QFileDialog,
            QProgressBar, QTabWidget, QMessageBox, QTableWidgetItem,
            QComboBox, QLineEdit, QTextEdit, QSplitter, QTableWidget
        )
        from PyQt6.QtGui import QColor
        QT_VERSION = 6
        print("Using PyQt6")
    except ImportError:
        raise ImportError("Neither PyQt5 nor PyQt6 is available. Please install PyQt5 or PyQt6.")

# Matplotlib backend configuration
import matplotlib
try:
    if QT_VERSION == 5:
        matplotlib.use('Qt5Agg')
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    else:
        matplotlib.use('QtAgg')
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    print(f"Matplotlib backend set to: {matplotlib.get_backend()}")
except Exception as e:
    print(f"Warning: Could not set Qt backend for matplotlib: {e}")
    # Fallback to a non-interactive backend
    matplotlib.use('Agg')
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    print("Using fallback matplotlib backend: Agg")

from matplotlib.figure import Figure

# Compatibility aliases for enums/constants
ALIGN_CENTER = getattr(Qt, 'AlignCenter', getattr(Qt, 'AlignmentFlag').AlignCenter)
HORIZONTAL = getattr(Qt, 'Horizontal', getattr(Qt, 'Orientation').Horizontal)
VERTICAL = getattr(Qt, 'Vertical', getattr(Qt, 'Orientation').Vertical)
LIGHT_GRAY = getattr(Qt, 'lightGray', getattr(Qt, 'GlobalColor').lightGray)

sys.path.append(os.path.join(os.path.dirname(__file__), ''))
from decov_function import ThermalAnalysisProcessor, setup_fonts, setup_plot_formatting


class ThermalAnalysisView(QMainWindow):
    """热分析系统主窗口类"""
    
    def __init__(self):
        super().__init__()
        # 初始化属性
        self.ambient_temp = 0.00655085  # 默认环境温度
        self.ploss = 1.0  # 默认损耗功率
        self.delta_z = 0.05  # 默认对数间隔
        self.num_iterations = 500  # 默认迭代次数
        self.discrete_order = None  # 用户必须填写离散阶数
        self.precision = 'float64'  # 默认计算精度
        self.results = {}  # 初始化结果字典
        self.data_mode = 'Original Temperature Data'  # 默认数据输入模式

        # 创建处理器实例
        self.processor = ThermalAnalysisProcessor(precision=self.precision)
        # 注意：discrete_order将在用户输入后设置

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
        title_label.setAlignment(ALIGN_CENTER)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; margin: 10px 0;")

        subtitle_label = QLabel("Thermal Resistance Extraction and Analysis Based on Structure Function Method")
        subtitle_label.setAlignment(ALIGN_CENTER)
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

        # 数据输入模式选择
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Data Input Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(['Original Temperature Data', 'Time Constant Spectrum Data'])
        self.mode_combo.currentTextChanged.connect(self.update_mode)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()

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
        self.ploss_slider = QSlider(HORIZONTAL)
        self.ploss_slider.setRange(1, 100)
        self.ploss_slider.setValue(10)
        self.ploss_value = QLabel("1.0")
        ploss_layout.addWidget(self.ploss_slider)
        ploss_layout.addWidget(self.ploss_value)

        # 环境温度
        ambient_layout = QVBoxLayout()
        ambient_layout.addWidget(QLabel("Ambient Temperature (°C)"))
        self.ambient_input = QLineEdit()
        self.ambient_input.setPlaceholderText("Enter ambient temperature (e.g., 25.0)")
        self.ambient_input.setText("0.00655085")  # 设置默认值
        self.ambient_input.setToolTip("Enter ambient temperature in Celsius. Values < 1.0 will use float64 precision, values ≥ 1.0 will use float32 precision.")
        self.ambient_input.textChanged.connect(self.update_ambient_temp)
        ambient_layout.addWidget(self.ambient_input)

        # 对数间隔
        delta_z_layout = QVBoxLayout()
        delta_z_layout.addWidget(QLabel("Log Interval Δz"))
        self.delta_z_slider = QSlider(HORIZONTAL)
        self.delta_z_slider.setRange(1, 100)
        self.delta_z_slider.setValue(5)
        self.delta_z_value = QLabel("0.05")
        delta_z_layout.addWidget(self.delta_z_slider)
        delta_z_layout.addWidget(self.delta_z_value)

        # 计算精度
        precision_layout = QVBoxLayout()
        precision_layout.addWidget(QLabel("Calculation Precision"))
        self.precision_combo = QComboBox()
        self.precision_combo.addItems(['float64', 'float32'])
        self.precision_combo.setCurrentText(self.precision)
        self.precision_combo.currentTextChanged.connect(self.update_precision)
        precision_layout.addWidget(self.precision_combo)

        # 离散阶数
        discrete_order_layout = QVBoxLayout()
        discrete_order_layout.addWidget(QLabel("Discrete Order"))
        self.discrete_order_input = QLineEdit()
        self.discrete_order_input.setPlaceholderText("Enter discrete order (e.g., 30)")
        self.discrete_order_input.setText("")  # 初始为空，要求用户填写
        self.discrete_order_input.textChanged.connect(self.update_discrete_order)
        discrete_order_layout.addWidget(self.discrete_order_input)

        param_layout.addLayout(ploss_layout)
        param_layout.addLayout(ambient_layout)
        param_layout.addLayout(delta_z_layout)
        param_layout.addLayout(precision_layout)
        param_layout.addLayout(discrete_order_layout)

        # 分析按钮
        self.analyze_btn = QPushButton("Start Analysis")
        self.analyze_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; "
            "font-size: 14pt; padding: 10px; border-radius: 8px;"
        )
        self.analyze_btn.clicked.connect(self.run_analysis)

        # 传递函数结果操作按钮
        transfer_function_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export Results")
        self.export_btn.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; "
            "font-size: 12pt; padding: 8px; border-radius: 6px;"
        )
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)  # 初始状态禁用
        
        self.import_btn = QPushButton("Import Results")
        self.import_btn.setStyleSheet(
            "background-color: #FF9800; color: white; font-weight: bold; "
            "font-size: 12pt; padding: 8px; border-radius: 6px;"
        )
        self.import_btn.clicked.connect(self.import_results)
        
        transfer_function_layout.addWidget(self.export_btn)
        transfer_function_layout.addWidget(self.import_btn)
        transfer_function_layout.addStretch()

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(False)

        control_layout.addLayout(mode_layout)
        control_layout.addLayout(file_layout)
        control_layout.addLayout(param_layout)
        control_layout.addWidget(self.analyze_btn)
        control_layout.addLayout(transfer_function_layout)
        control_layout.addWidget(self.progress_bar)

        # 图表区域
        self.tab_widget = QTabWidget()

        # 创建图表标签页
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tab4 = QWidget()
        self.tab5 = QWidget()

        self.tab_widget.addTab(self.tab1, "Original Data")
        self.tab_widget.addTab(self.tab2, "Log Interpolation")
        self.tab_widget.addTab(self.tab3, "Time Constant Spectrum")
        self.tab_widget.addTab(self.tab4, "Structure Function")
        self.tab_widget.addTab(self.tab5, "Transfer Function Results")

        # 设置标签页布局
        self.setup_tab1()
        self.setup_tab2()
        self.setup_tab3()
        self.setup_tab4()
        self.setup_tab5()

        # 主布局
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.tab_widget, 1)

        # 连接信号
        self.ploss_slider.valueChanged.connect(self.update_ploss_value)
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

    def setup_tab5(self):
        """设置传递函数结果标签页"""
        layout = QVBoxLayout(self.tab5)

        # 创建分割器
        splitter = QSplitter(VERTICAL)
        
        # 上半部分：参数表格
        self.transfer_table = QTableWidget()
        self.transfer_table.setColumnCount(4)
        self.transfer_table.setHorizontalHeaderLabels(['Parameter', 'Value', 'Unit', 'Description'])
        self.transfer_table.horizontalHeader().setStretchLastSection(True)
        
        # 下半部分：详细信息文本
        self.transfer_text = QTextEdit()
        self.transfer_text.setReadOnly(True)
        self.transfer_text.setMaximumHeight(200)
        
        splitter.addWidget(self.transfer_table)
        splitter.addWidget(self.transfer_text)
        splitter.setSizes([400, 200])
        
        layout.addWidget(splitter)

    def browse_file(self):
        """浏览文件"""
        if self.data_mode == 'Time Constant Spectrum Data':
            title = "Select Time Constant Spectrum Data File"
            file_filter = "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
        else:
            title = "Select Temperature Data File"
            file_filter = "Excel Files (*.xlsx *.xls);;CSV Files (*.csv);;All Files (*)"
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, title, "", file_filter
        )

        if file_path:
            self.file_label.setText(os.path.basename(file_path))
            self.file_path = file_path

    def update_ploss_value(self, value):
        """更新损耗功率值"""
        ploss = value / 10.0
        self.ploss_value.setText(f"{ploss:.1f}")
        self.processor.ploss = ploss

    def update_ambient_temp(self, text):
        """更新环境温度值并自动切换计算精度"""
        try:
            if text.strip():
                # 解析输入的温度值
                ambient_temp = float(text.strip())
                
                # 验证温度值的合理性
                if ambient_temp < -273.15:
                    QMessageBox.warning(self, "Invalid Temperature", 
                                      "Temperature cannot be below absolute zero (-273.15°C)")
                    return
                elif ambient_temp > 1000:
                    QMessageBox.warning(self, "Invalid Temperature", 
                                      "Temperature seems too high (>1000°C). Please check your input.")
                    return
                
                self.ambient_temp = ambient_temp
                
                # 自动切换计算精度
                if ambient_temp < 1.0:
                    # 对于很小的温度值，使用float64精度
                    new_precision = 'float64'
                else:
                    # 对于正常温度值，使用float32精度
                    new_precision = 'float32'
                
                # 如果精度发生变化，更新处理器
                if new_precision != self.precision:
                    self.precision = new_precision
                    self.precision_combo.setCurrentText(new_precision)
                    self.processor = ThermalAnalysisProcessor(precision=self.precision)
                    # 同步环境温度到处理器
                    self.processor.ambient_temp = self.ambient_temp
                    print(f"环境温度: {ambient_temp}°C, 自动切换计算精度为: {new_precision}")
                else:
                    # 同步环境温度到处理器
                    self.processor.ambient_temp = self.ambient_temp
                    print(f"环境温度已更新为: {ambient_temp}°C")
                    
            else:
                # 如果输入为空，重置为默认值
                self.ambient_temp = 0.00655085
                self.processor.ambient_temp = self.ambient_temp
                print("环境温度输入为空，使用默认值: 0.00655085°C")
                
        except ValueError:
            # 如果输入格式错误，显示警告但保持当前值
            print("环境温度输入格式错误，请输入有效的数值")
            QMessageBox.warning(self, "Invalid Input", 
                              "Please enter a valid number for ambient temperature.")
        except Exception as e:
            print(f"更新环境温度时发生错误: {e}")
            QMessageBox.warning(self, "Error", 
                              f"An error occurred while updating ambient temperature: {str(e)}")

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
        # 同步环境温度到处理器
        self.processor.ambient_temp = self.ambient_temp
        print(f"计算精度已更新为: {precision}")

    def update_discrete_order(self, text):
        """更新离散阶数"""
        try:
            if text.strip():
                self.discrete_order = int(text.strip())
                self.processor.discrete_order = self.discrete_order
                print(f"离散阶数已更新为: {self.discrete_order}")
            else:
                self.discrete_order = None
                print("离散阶数未设置")
        except ValueError:
            self.discrete_order = None
            print("离散阶数格式错误，请输入整数")

    def update_mode(self, mode):
        """更新数据输入模式"""
        self.data_mode = mode
        print(f"数据输入模式已更新为: {mode}")
        # 根据模式更新文件选择提示
        if mode == 'Time Constant Spectrum Data':
            self.file_label.setText("Select time constant spectrum data file (Time, R(z))")
        else:
            self.file_label.setText("No file selected")

    def run_analysis(self):
        """运行分析"""
        if not hasattr(self, 'file_path'):
            QMessageBox.warning(self, "No File Selected", "Please select a data file first.")
            return

        # 验证离散阶数是否已设置
        if self.discrete_order is None:
            QMessageBox.warning(self, "Discrete Order Required", 
                              "Please enter a discrete order value before running analysis.")
            return

        # 验证离散阶数是否为正整数
        if self.discrete_order <= 0:
            QMessageBox.warning(self, "Invalid Discrete Order", 
                              "Discrete order must be a positive integer.")
            return

        # 更新进度条
        self.progress_bar.setValue(10)

        # 根据数据输入模式选择分析方法
        if self.data_mode == 'Time Constant Spectrum Data':
            # 从时间常数谱数据开始分析
            success = self.processor.analysis_from_time_constant_spectrum(self.file_path)
        else:
            # 从原始温度数据开始完整分析
            # 更新处理器的环境温度
            self.processor.ambient_temp = self.ambient_temp
            success = self.processor.full_analysis(self.file_path)

        if success:
            self.progress_bar.setValue(100)
            self.plot_results()
            self.update_transfer_function_display()
            self.export_btn.setEnabled(True)  # 启用导出按钮
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

            # 自动调整y轴精度
            self._adjust_y_axis_precision(ax, self.processor.Tj)

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

        # 检查是否有完整的分析数据
        has_full_data = ('z_fft' in self.processor.results and 'az_fft' in self.processor.results)
        
        if has_full_data:
            # 完整分析数据
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
            # 自动调整y轴精度
            self._adjust_y_axis_precision(ax1, self.processor.results['az_fft'])
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
                # 自动调整y轴精度
                self._adjust_y_axis_precision(ax2, self.processor.results['az_bayesian'])
                ax2.grid(True, linestyle='--', alpha=0.7)
        elif 'z_bayesian' in self.processor.results and 'R' in self.processor.results:
            # 从时间常数谱数据开始的情况
            ax = self.fig2.add_subplot(111)
            ax.plot(self.processor.results['z_bayesian'], self.processor.results['R'], 'g-', linewidth=2)
            try:
                ax.set_xlabel('z = ln(t)', fontsize=12)
                ax.set_ylabel('R(z)', fontsize=12)
                ax.set_title('Imported Time Constant Spectrum on Log Time Axis', fontsize=14, fontweight='bold')
            except:
                ax.set_xlabel('z = ln(t)', fontsize=12)
                ax.set_ylabel('R(z)', fontsize=12)
                ax.set_title('Imported Time Constant Spectrum', fontsize=14, fontweight='bold')
            setup_plot_formatting(ax)
            # 自动调整y轴精度
            self._adjust_y_axis_precision(ax, self.processor.results['R'])
            ax.grid(True, linestyle='--', alpha=0.7)
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

        if ('t_bayesian' in self.processor.results and 'R' in self.processor.results):
            # 检查是否有完整的分析数据
            has_full_data = ('az_bayesian' in self.processor.results and 
                           'da_dz_bayesian' in self.processor.results)
            
            if has_full_data:
                # 创建三个子图（完整分析）
                ax1 = self.fig3.add_subplot(311)
                ax2 = self.fig3.add_subplot(312)
                ax3 = self.fig3.add_subplot(313)

                # 为第1图设置横轴科学计数法格式
                def sci_formatter_x(val, pos=None):
                    """科学计数法格式化器，与结构函数纵轴格式一致"""
                    s = f"{val:.1e}"
                    s = s.replace('−', '-')  # Replace Unicode minus sign
                    s = s.replace('e-', '×10⁻')  # Replace scientific notation
                    s = s.replace('e+', '×10⁺')  # Replace scientific notation
                    return s
                
                import matplotlib.ticker as ticker
                
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
                # 在setup_plot_formatting之后重新设置横轴格式，确保不被覆盖
                ax1.xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_x))
                # 自动调整y轴精度
                self._adjust_y_axis_precision(ax1, az_bayesian)
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
                # 自动调整y轴精度（使用平滑后的导数数据）
                self._adjust_y_axis_precision(ax2, da_dz_bayesian_smoothed)
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
                # 在setup_plot_formatting之后重新设置横轴格式，确保不被覆盖
                ax3.xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_x))
                # 自动调整y轴精度
                self._adjust_y_axis_precision(ax3, R)
                ax3.grid(True, linestyle='--', alpha=0.7)
            else:
                # 只有时间常数谱数据（从文件导入）
                ax = self.fig3.add_subplot(111)
                
                # 为横轴设置科学计数法格式
                def sci_formatter_x(val, pos=None):
                    s = f"{val:.1e}"
                    s = s.replace('−', '-')
                    s = s.replace('e-', '×10⁻')
                    s = s.replace('e+', '×10⁺')
                    return s
                
                import matplotlib.ticker as ticker
                
                # 绘制时间常数谱
                t_bayesian = self.processor.results['t_bayesian']
                R = self.processor.results['R']
                ax.semilogx(t_bayesian, R, 'g-', linewidth=2)
                try:
                    ax.set_title('Imported Time Constant Spectrum R(z)', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time (s)', fontsize=10)
                    ax.set_ylabel('R(z)', fontsize=10)
                except:
                    ax.set_title('Imported Time Constant Spectrum R(z)', fontsize=12, fontweight='bold')
                    ax.set_xlabel('Time (s)', fontsize=10)
                    ax.set_ylabel('R(z)', fontsize=10)
                
                setup_plot_formatting(ax)
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(sci_formatter_x))
                # 自动调整y轴精度
                self._adjust_y_axis_precision(ax, R)
                ax.grid(True, linestyle='--', alpha=0.7)
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
                ax1.set_xlabel('Integral thermal resistance ∑Rth (K/W)', fontsize=10)
                ax1.set_ylabel('Integral thermal capacity ∑Cth (Ws/K)', fontsize=10)
                ax1.grid(True, linestyle='--', alpha=0.7)
                # 设置纵坐标最大值为10^65
                ax1.set_ylim(bottom=ax1.get_ylim()[0], top=1e8)
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

    def _adjust_y_axis_precision(self, ax, data):
        """
        根据数据范围自动调整y轴精度，特别优化对数函数图表的显示
        动态扩展小数位直到能够区分y轴坐标
        
        Args:
            ax: matplotlib轴对象
            data: 数据数组
        """
        try:
            import matplotlib.ticker as ticker
            
            # 过滤有效数据
            valid_data = data[np.isfinite(data)]
            if len(valid_data) == 0:
                return
            
            # 计算数据范围
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            data_range = data_max - data_min
            
            # 计算相对精度（数据范围相对于数据大小的比例）
            data_magnitude = max(abs(data_min), abs(data_max))
            if data_magnitude > 0:
                relative_range = data_range / data_magnitude
            else:
                relative_range = data_range
            
            # 动态确定所需的小数位数
            def find_required_precision(values, max_precision=15):
                """
                找到能够区分所有值所需的最小小数位数
                """
                if len(values) <= 1:
                    return 2
                
                # 尝试不同的小数位数
                for precision in range(2, max_precision + 1):
                    formatted_values = [f'{v:.{precision}e}' for v in values]
                    if len(set(formatted_values)) == len(values):
                        return precision
                
                return max_precision
            
            # 获取y轴刻度位置
            ax.locator_params(axis='y', nbins=8)
            y_ticks = ax.get_yticks()
            
            # 过滤有效的刻度值
            valid_ticks = y_ticks[np.isfinite(y_ticks)]
            if len(valid_ticks) > 1:
                # 计算刻度值范围
                tick_min = np.min(valid_ticks)
                tick_max = np.max(valid_ticks)
                tick_range = tick_max - tick_min
                
                # 如果刻度范围很小，需要更多精度
                if tick_range < 1e-10:
                    required_precision = find_required_precision(valid_ticks, 15)
                elif tick_range < 1e-6:
                    required_precision = find_required_precision(valid_ticks, 12)
                elif tick_range < 1e-3:
                    required_precision = find_required_precision(valid_ticks, 8)
                else:
                    required_precision = 4
            else:
                # 如果没有有效刻度，使用数据范围来确定精度
                if data_range < 1e-10:
                    required_precision = 12
                elif data_range < 1e-6:
                    required_precision = 8
                elif data_range < 1e-3:
                    required_precision = 6
                else:
                    required_precision = 4
            
            # 根据数据范围和相对精度选择格式
            if data_range < 1e-15 or relative_range < 1e-12:
                # 极小范围数据，使用动态精度科学计数法
                def formatter(x, pos):
                    if abs(x) < 1e-10:
                        return f'{x:.{max(required_precision, 8)}e}'
                    else:
                        return f'{x:.{max(required_precision, 6)}e}'
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            elif data_range < 1e-12 or relative_range < 1e-9:
                # 极小范围数据，使用动态精度科学计数法
                def formatter(x, pos):
                    if abs(x) < 1e-8:
                        return f'{x:.{max(required_precision, 6)}e}'
                    else:
                        return f'{x:.{max(required_precision, 4)}e}'
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            elif data_range < 1e-9 or relative_range < 1e-6:
                # 小范围数据，使用动态精度科学计数法
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: f'{x:.{max(required_precision, 4)}e}'))
            elif data_range < 1e-6 or relative_range < 1e-3:
                # 小范围数据，使用动态精度科学计数法
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: f'{x:.{max(required_precision, 3)}e}'))
            elif data_range < 1e-3 or relative_range < 1e-1:
                # 中等范围数据，使用动态精度
                if data_magnitude < 1:
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                        lambda x, p: f'{x:.{max(required_precision, 2)}e}'))
                else:
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                        lambda x, p: f'{x:.{max(required_precision, 6)}f}'))
            elif data_range < 1 or relative_range < 0.1:
                # 较大范围数据，使用动态精度固定小数
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                    lambda x, p: f'{x:.{max(required_precision, 4)}f}'))
            else:
                # 大范围数据，使用科学计数法
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
 
            # 设置y轴刻度数量，确保有足够的精度
            # 对于小范围数据，增加刻度数量以显示更多细节
            if data_range < 1e-6:
                ax.locator_params(axis='y', nbins=10)  # 更多刻度
            elif data_range < 1e-3:
                ax.locator_params(axis='y', nbins=8)   # 标准刻度
            else:
                ax.locator_params(axis='y', nbins=6)   # 较少刻度
            
            # 验证刻度是否能够区分
            self._verify_tick_distinction(ax, data)
            
        except Exception as e:
            print(f"调整y轴精度时出错: {e}")
            # 如果出错，使用默认的科学计数法
            try:
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            except:
                pass

    def _verify_tick_distinction(self, ax, data):
        """
        验证y轴刻度是否能够区分，如果不能则增加精度
        
        Args:
            ax: matplotlib轴对象
            data: 数据数组
        """
        try:
            import matplotlib.ticker as ticker
            
            # 获取当前刻度标签
            y_ticks = ax.get_yticks()
            valid_ticks = y_ticks[np.isfinite(y_ticks)]
            
            if len(valid_ticks) <= 1:
                return
            
            # 获取当前格式化器
            formatter = ax.yaxis.get_major_formatter()
            
            # 检查刻度标签是否重复
            tick_labels = [formatter.format_data(tick) for tick in valid_ticks]
            unique_labels = set(tick_labels)
            
            # 如果标签重复，增加精度
            if len(unique_labels) < len(valid_ticks):
                # 找到当前使用的精度
                current_precision = 4
                if hasattr(formatter, 'func') and hasattr(formatter.func, '__code__'):
                    # 尝试从格式化函数中提取精度
                    import re
                    func_str = str(formatter.func)
                    match = re.search(r'\.(\d+)e', func_str)
                    if match:
                        current_precision = int(match.group(1))
                
                # 逐步增加精度直到能够区分
                for precision in range(current_precision + 1, 20):
                    def new_formatter(x, pos):
                        if 'e' in str(formatter.func):
                            return f'{x:.{precision}e}'
                        else:
                            return f'{x:.{precision}f}'
                    
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(new_formatter))
                    
                    # 重新检查是否能够区分
                    new_tick_labels = [new_formatter(tick, None) for tick in valid_ticks]
                    new_unique_labels = set(new_tick_labels)
                    
                    if len(new_unique_labels) == len(valid_ticks):
                        break
                        
        except Exception as e:
            print(f"验证刻度区分时出错: {e}")

    def export_results(self):
        """导出传递函数计算结果"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transfer Function Results", "", 
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            if not file_path.endswith('.xlsx'):
                file_path += '.xlsx'
            
            success = self.processor.export_transfer_function_results(file_path)
            if success:
                QMessageBox.information(self, "Export Success", 
                                      f"Transfer function results exported to:\n{file_path}")
            else:
                QMessageBox.warning(self, "Export Failed", 
                                   "Failed to export transfer function results.")

    def import_results(self):
        """导入传递函数计算结果"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Transfer Function Results", "", 
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            success = self.processor.import_transfer_function_results(file_path)
            if success:
                # 更新UI中的离散阶数显示
                if self.processor.discrete_order is not None:
                    self.discrete_order_input.setText(str(self.processor.discrete_order))
                    self.discrete_order = self.processor.discrete_order
                else:
                    self.discrete_order_input.setText("")
                    self.discrete_order = None
                
                self.update_transfer_function_display()
                self.plot_structure_functions()  # 重新绘制结构函数
                QMessageBox.information(self, "Import Success", 
                                      f"Transfer function results imported from:\n{file_path}")
            else:
                QMessageBox.warning(self, "Import Failed", 
                                   "Failed to import transfer function results.")

    def update_transfer_function_display(self):
        """更新传递函数结果显示"""
        
        # 清空表格
        self.transfer_table.setRowCount(0)
        
        # 准备显示数据
        display_data = []
        
        # Foster网络参数
        if 'fosterRth' in self.processor.results and 'fosterCth' in self.processor.results:
            foster_rth = self.processor.results['fosterRth']
            foster_cth = self.processor.results['fosterCth']
            
            if len(foster_rth) > 0:
                display_data.append(('Foster Network Parameters', '', '', ''))
                display_data.append(('Number of Parameters', str(len(foster_rth)), '', ''))
                display_data.append(('Total Rth', f"{np.sum(foster_rth):.6e}", 'K/W', 'Total thermal resistance'))
                display_data.append(('Total Cth', f"{np.sum(foster_cth):.6e}", 'Ws/K', 'Total thermal capacity'))
                display_data.append(('Min Rth', f"{np.min(foster_rth):.6e}", 'K/W', 'Minimum thermal resistance'))
                display_data.append(('Max Rth', f"{np.max(foster_rth):.6e}", 'K/W', 'Maximum thermal resistance'))
                display_data.append(('Min Cth', f"{np.min(foster_cth):.6e}", 'Ws/K', 'Minimum thermal capacity'))
                display_data.append(('Max Cth', f"{np.max(foster_cth):.6e}", 'Ws/K', 'Maximum thermal capacity'))
        
        # Cauer网络参数
        if 'cauerRth' in self.processor.results and 'cauerCth' in self.processor.results:
            cauer_rth = self.processor.results['cauerRth']
            cauer_cth = self.processor.results['cauerCth']
            
            if len(cauer_rth) > 0:
                display_data.append(('', '', '', ''))  # 空行
                display_data.append(('Cauer Network Parameters', '', '', ''))
                display_data.append(('Number of Parameters', str(len(cauer_rth)), '', ''))
                display_data.append(('Total Rth', f"{np.sum(cauer_rth):.6e}", 'K/W', 'Total thermal resistance'))
                display_data.append(('Total Cth', f"{np.sum(cauer_cth):.6e}", 'Ws/K', 'Total thermal capacity'))
                display_data.append(('Min Rth', f"{np.min(cauer_rth):.6e}", 'K/W', 'Minimum thermal resistance'))
                display_data.append(('Max Rth', f"{np.max(cauer_rth):.6e}", 'K/W', 'Maximum thermal resistance'))
                display_data.append(('Min Cth', f"{np.min(cauer_cth):.6e}", 'Ws/K', 'Minimum thermal capacity'))
                display_data.append(('Max Cth', f"{np.max(cauer_cth):.6e}", 'Ws/K', 'Maximum thermal capacity'))
        
        # 结构函数信息
        if 'cumulative_Rth' in self.processor.results and 'cumulative_Cth' in self.processor.results:
            cumulative_rth = self.processor.results['cumulative_Rth']
            cumulative_cth = self.processor.results['cumulative_Cth']
            
            if len(cumulative_rth) > 0:
                display_data.append(('', '', '', ''))  # 空行
                display_data.append(('Structure Function Parameters', '', '', ''))
                display_data.append(('Integral Data Points', str(len(cumulative_rth)), '', ''))
                display_data.append(('Final Rth', f"{cumulative_rth[-1]:.6e}", 'K/W', 'Final cumulative thermal resistance'))
                display_data.append(('Final Cth', f"{cumulative_cth[-1]:.6e}", 'Ws/K', 'Final cumulative thermal capacity'))
        
        if 'differential_Rth' in self.processor.results and 'differential_Cth' in self.processor.results:
            differential_rth = self.processor.results['differential_Rth']
            differential_cth = self.processor.results['differential_Cth']
            
            if len(differential_rth) > 0:
                display_data.append(('Differential Data Points', str(len(differential_rth)), '', ''))
                display_data.append(('Min Diff Rth', f"{np.min(differential_rth):.6e}", 'K/W', 'Minimum differential thermal resistance'))
                display_data.append(('Max Diff Rth', f"{np.max(differential_rth):.6e}", 'K/W', 'Maximum differential thermal resistance'))
                display_data.append(('Min Diff Cth', f"{np.min(differential_cth):.6e}", 'Ws/K', 'Minimum differential thermal capacity'))
                display_data.append(('Max Diff Cth', f"{np.max(differential_cth):.6e}", 'Ws/K', 'Maximum differential thermal capacity'))
        
        # 计算参数
        display_data.append(('', '', '', ''))  # 空行
        display_data.append(('Calculation Parameters', '', '', ''))
        display_data.append(('Precision', self.processor.precision, '', 'Calculation precision'))
        discrete_order_display = str(self.processor.discrete_order) if self.processor.discrete_order is not None else "Not set"
        display_data.append(('Discrete Order', discrete_order_display, '', 'Discrete time constant spectrum order'))
        display_data.append(('Delta z', f"{self.processor.delta_z:.3f}", '', 'Log interval'))
        
        # 填充表格
        self.transfer_table.setRowCount(len(display_data))
        for i, (param, value, unit, desc) in enumerate(display_data):
            self.transfer_table.setItem(i, 0, QTableWidgetItem(param))
            self.transfer_table.setItem(i, 1, QTableWidgetItem(value))
            self.transfer_table.setItem(i, 2, QTableWidgetItem(unit))
            self.transfer_table.setItem(i, 3, QTableWidgetItem(desc))
            
            # 设置标题行样式
            if param in ['Foster Network Parameters', 'Cauer Network Parameters', 'Structure Function Parameters', 'Calculation Parameters']:
                for j in range(4):
                    item = self.transfer_table.item(i, j)
                    if item:
                        item.setBackground(LIGHT_GRAY)
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
        
        # 调整列宽
        self.transfer_table.resizeColumnsToContents()
        
        # 更新详细信息文本
        self.update_transfer_function_text()

    def update_transfer_function_text(self):
        """更新传递函数详细信息文本"""
        text = "Transfer Function Analysis Results\n"
        text += "=" * 50 + "\n\n"
        
        # Foster网络详细信息
        if 'fosterRth' in self.processor.results and 'fosterCth' in self.processor.results:
            foster_rth = self.processor.results['fosterRth']
            foster_cth = self.processor.results['fosterCth']
            
            text += "Foster Network Parameters:\n"
            text += f"  Number of parameters: {len(foster_rth)}\n"
            text += f"  Total thermal resistance: {np.sum(foster_rth):.6e} K/W\n"
            text += f"  Total thermal capacity: {np.sum(foster_cth):.6e} Ws/K\n\n"
            
            text += "Individual Parameters:\n"
            for i in range(min(len(foster_rth), 10)):  # 只显示前10个参数
                text += f"  R{i+1}: {foster_rth[i]:.6e} K/W, C{i+1}: {foster_cth[i]:.6e} Ws/K, τ{i+1}: {foster_rth[i]*foster_cth[i]:.6e} s\n"
            
            if len(foster_rth) > 10:
                text += f"  ... and {len(foster_rth) - 10} more parameters\n"
            text += "\n"
        
        # Cauer网络详细信息
        if 'cauerRth' in self.processor.results and 'cauerCth' in self.processor.results:
            cauer_rth = self.processor.results['cauerRth']
            cauer_cth = self.processor.results['cauerCth']
            
            text += "Cauer Network Parameters:\n"
            text += f"  Number of parameters: {len(cauer_rth)}\n"
            text += f"  Total thermal resistance: {np.sum(cauer_rth):.6e} K/W\n"
            text += f"  Total thermal capacity: {np.sum(cauer_cth):.6e} Ws/K\n\n"
            
            text += "Individual Parameters:\n"
            for i in range(min(len(cauer_rth), 10)):  # 只显示前10个参数
                text += f"  R{i+1}: {cauer_rth[i]:.6e} K/W, C{i+1}: {cauer_cth[i]:.6e} Ws/K, τ{i+1}: {cauer_rth[i]*cauer_cth[i]:.6e} s\n"
            
            if len(cauer_rth) > 10:
                text += f"  ... and {len(cauer_rth) - 10} more parameters\n"
            text += "\n"
        
        # 结构函数信息
        if 'cumulative_Rth' in self.processor.results and 'cumulative_Cth' in self.processor.results:
            cumulative_rth = self.processor.results['cumulative_Rth']
            cumulative_cth = self.processor.results['cumulative_Cth']
            
            text += "Structure Function Information:\n"
            text += f"  Integral data points: {len(cumulative_rth)}\n"
            text += f"  Final cumulative Rth: {cumulative_rth[-1]:.6e} K/W\n"
            text += f"  Final cumulative Cth: {cumulative_cth[-1]:.6e} Ws/K\n\n"
        
        self.transfer_text.setText(text)


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
    sys.exit(QApplication.instance().exec())
