import sys
from cmath import log

import numpy as np
from scipy.linalg import solve
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSlider, QLabel, QPushButton, QGroupBox, QSizePolicy)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import random
from scipy.interpolate import interp1d

class BayesianDeconvolutionModel:
    def __init__(self):
        self.signal_length = 100
        self.kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        self.original_signal = None
        self.convolved = None
        self.observed_signal = None
        self.reconstructed = None
        self.direct_reverse = None
        self.noise_level = 0.2
        self.prior_strength = 0.05

    def generate_signal(self):
        signal = np.zeros(self.signal_length)
        for i in range(3):
            pos = 20 + i * 30
            amp = 0.5 + random.random() * 0.5
            for j in range(-10, 11):
                idx = pos + j
                if 0 <= idx < self.signal_length:
                    signal[idx] += amp * np.exp(-0.5 * (j / 3) ** 2)
        return signal

    def logarithmic_interpolation(self, t0, zth_transient):
        """
        对时间轴取对数并进行插值计算
        参数:
        t0 -- 原始时间轴数组
        Zth_transient -- 对应的瞬态热阻抗值数组
        返回:
        z_fft -- 对数时间轴
        az_fft -- 插值后的瞬态热阻抗值
        """
        # 1. 时间轴取对数: z = ln(t)
        z_fft = np.log(t0)
        # 2. 创建插值函数
        # 使用样条插值方法，在原始时间点(t0)和对应的瞬态热阻抗值(Zth_transient)之间建立关系
        interp_func = interp1d(t0, zth_transient, kind='cubic', fill_value='extrapolate')
        # 3. 计算 a(z) = Zth(t=exp(z))
        # 在对数时间轴上计算对应的瞬态热阻抗值
        az_fft = interp_func(np.exp(z_fft))
        return z_fft, az_fft

    def convolve(self, signal, kernel):
        result = np.zeros(len(signal) + len(kernel) - 1)
        for i in range(len(signal)):
            for j in range(len(kernel)):
                result[i + j] += signal[i] * kernel[j]
        return result[:len(signal)]

    def bayesian_deconvolution(self, observed, kernel, prior_strength):
        signal_length = len(observed)
        # 构建卷积矩阵 H
        H = np.zeros((signal_length, signal_length))
        kernel_len = len(kernel)

        for i in range(signal_length):
            for j in range(kernel_len):
                col_idx = i - j
                if 0 <= col_idx < signal_length:
                    H[i, col_idx] = kernel[j]

        Ht = H.T
        HtH = Ht.dot(H)
        lambda_sq = prior_strength ** 2
        reg_matrix = HtH + lambda_sq * np.identity(signal_length)
        Hty = Ht.dot(observed)

        return solve(reg_matrix, Hty, assume_a='sym')

    def run_simulation(self, noise_level=None, prior_strength=None):
        # 如果参数未提供，使用当前值
        if noise_level is None:
            noise_level = self.noise_level
        if prior_strength is None:
            prior_strength = self.prior_strength

        # 更新模型参数
        self.noise_level = noise_level
        self.prior_strength = prior_strength

        # 生成信号
        self.original_signal = self.generate_signal()

        # 卷积操作
        self.convolved = self.convolve(self.original_signal, self.kernel)

        # 添加噪声
        noise = np.random.uniform(-1, 1, len(self.convolved)) * noise_level
        self.observed_signal = self.convolved + noise

        # 贝叶斯反卷积
        self.reconstructed = self.bayesian_deconvolution(
            self.observed_signal,
            self.kernel,
            prior_strength
        )

        return (
            self.original_signal,
            self.observed_signal,
            self.reconstructed
        )


class BayesianDeconvolutionView(QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.init_ui()
        self.setup_connections()

    def init_ui(self):
        self.setWindowTitle('贝叶斯反卷积噪声误差分析')
        self.setGeometry(100, 100, 1200, 800)

        # 主布局
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 标题
        title_layout = QVBoxLayout()
        title_label = QLabel("贝叶斯反卷积噪声误差分析")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 24pt; font-weight: bold; margin: 20px 0;")

        subtitle_label = QLabel("探索噪声水平如何影响反卷积结果稳定性")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("font-size: 14pt; color: #666; margin-bottom: 30px;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        main_layout.addLayout(title_layout)

        # 控制面板
        control_group = QGroupBox()
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)
        control_group.setStyleSheet(
            "QGroupBox {background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #6a11cb, stop:1 #2575fc); border-radius: 15px;"
            "border: none; padding: 15px;}"
        )

        # 噪声控制滑块
        noise_layout = QHBoxLayout()
        noise_label = QLabel("噪声水平 σ")
        noise_label.setStyleSheet("color: white; font-size: 12pt; font-weight: bold;")
        self.noise_slider = QSlider(Qt.Horizontal)
        self.noise_slider.setRange(1, 100)
        self.noise_slider.setValue(20)
        self.noise_value = QLabel("0.2")
        self.noise_value.setStyleSheet(
            "background-color: white; color: black; padding: 3px 8px;"
            "border-radius: 10px; font-weight: bold; min-width: 40px;"
        )
        noise_layout.addWidget(noise_label)
        noise_layout.addWidget(self.noise_slider)
        noise_layout.addWidget(self.noise_value)

        # 先验强度滑块
        prior_layout = QHBoxLayout()
        prior_label = QLabel("先验强度 λ")
        prior_label.setStyleSheet("color: white; font-size: 12pt; font-weight: bold;")
        self.prior_slider = QSlider(Qt.Horizontal)
        self.prior_slider.setRange(1, 500)
        self.prior_slider.setValue(50)
        self.prior_value = QLabel("0.05")
        self.prior_value.setStyleSheet(
            "background-color: white; color: black; padding: 3px 8px;"
            "border-radius: 10px; font-weight: bold; min-width: 40px;"
        )
        prior_layout.addWidget(prior_label)
        prior_layout.addWidget(self.prior_slider)
        prior_layout.addWidget(self.prior_value)

        # 运行按钮
        self.simulate_btn = QPushButton("运行贝叶斯反卷积模拟")
        self.simulate_btn.setStyleSheet(
            "background-color: white; color: #6a11cb; font-weight: bold; "
            "font-size: 14pt; padding: 12px; border-radius: 8px;"
        )

        control_layout.addLayout(noise_layout)
        control_layout.addLayout(prior_layout)
        control_layout.addWidget(self.simulate_btn)
        main_layout.addWidget(control_group)

        # 图例控制
        legend_layout = QHBoxLayout()
        legend_layout.setAlignment(Qt.AlignCenter)

        self.legend_btns = []
        colors = ['#6a11cb', '#2575fc', '#ff2e63']
        labels = ['原始信号', '含噪观测信号', '贝叶斯反卷积重建']

        for i, label in enumerate(labels):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(
                f"QPushButton {{background-color: {colors[i]}; color: white; "
                "padding: 6px 12px; border-radius: 15px; font-weight: bold;}}"
                "QPushButton:checked {opacity: 1;}"
                "QPushButton:!checked {opacity: 0.5;}"
            )
            legend_layout.addWidget(btn)
            self.legend_btns.append(btn)

        legend_container = QWidget()
        legend_container.setLayout(legend_layout)
        main_layout.addWidget(legend_container)

        # 图表区域
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left', "信号幅值")
        self.plot_widget.setLabel('bottom', "时间/样本点")

        # 创建图例
        self.plot_widget.addLegend(offset=(10, 10))

        # 初始化绘图项
        self.original_plot = self.plot_widget.plot(
            [], [],
            name="原始信号",
            pen=pg.mkPen(color='#6a11cb', width=3)
        )
        self.observed_plot = self.plot_widget.plot(
            [], [],
            name="含噪观测信号",
            pen=pg.mkPen(color='#2575fc', width=2, dash=[5, 5])
        )
        self.reconstructed_plot = self.plot_widget.plot(
            [], [],
            name="贝叶斯反卷积重建",
            pen=pg.mkPen(color='#ff2e63', width=3)
        )

        # 添加标题
        title = pg.TextItem("贝叶斯反卷积演示", color=(70, 70, 70), anchor=(0.5, 1))
        title.setFont(QApplication.font())
        # title.setFont("Arial")
        self.plot_widget.addItem(title)
        title.setPos(50, self.plot_widget.getViewBox().viewRect().y() - 20)

        main_layout.addWidget(self.plot_widget, 1)

    def setup_connections(self):
        """设置信号和槽连接"""
        self.simulate_btn.clicked.connect(self.handle_simulate)
        self.noise_slider.valueChanged.connect(self.update_noise_value)
        self.prior_slider.valueChanged.connect(self.update_prior_value)

        # 连接图例控制
        for btn in self.legend_btns:
            btn.toggled.connect(self.handle_legend_toggled)

    def handle_simulate(self):
        """处理运行模拟按钮点击"""
        # 从滑块获取当前值
        noise_level = self.noise_slider.value() / 100.0
        prior_strength = self.prior_slider.value() / 1000.0

        # 通知控制器运行模拟
        self.controller.run_simulation(noise_level, prior_strength)

        # 获取结果并更新图表
        original, observed, reconstructed = self.controller.get_results()
        self.update_plots(original, observed, reconstructed)

    def update_plots(self, original, observed, reconstructed):
        """更新图表数据"""
        x = np.arange(len(original))

        self.original_plot.setData(x, original)
        self.observed_plot.setData(x, observed)
        self.reconstructed_plot.setData(x, reconstructed)

        # 设置Y轴范围
        min_val = min(np.min(original), np.min(observed), np.min(reconstructed)) - 0.5
        max_val = max(np.max(original), np.max(observed), np.max(reconstructed)) + 0.5
        self.plot_widget.setYRange(min_val, max_val)

    def update_noise_value(self, value):
        """更新噪声值显示"""
        self.noise_value.setText(f"{value / 100:.2f}")

    def update_prior_value(self, value):
        """更新先验强度值显示"""
        self.prior_value.setText(f"{value / 1000:.3f}")

    def handle_legend_toggled(self, checked):
        """处理图例按钮切换"""
        # 确定哪个按钮发送了信号
        sender = self.sender()
        index = self.legend_btns.index(sender)

        # 应用可见性设置
        if index == 0:
            self.original_plot.setVisible(checked)
        elif index == 1:
            self.observed_plot.setVisible(checked)
        elif index == 2:
            self.reconstructed_plot.setVisible(checked)

        # 更新Y轴范围
        visible_plots = []
        if self.legend_btns[0].isChecked():
            visible_plots.append(self.original_plot)
        if self.legend_btns[1].isChecked():
            visible_plots.append(self.observed_plot)
        if self.legend_btns[2].isChecked():
            visible_plots.append(self.reconstructed_plot)

        if visible_plots:
            min_val = min(min(plot.yData) for plot in visible_plots) - 0.5
            max_val = max(max(plot.yData) for plot in visible_plots) + 0.5
            self.plot_widget.setYRange(min_val, max_val)


class BayesianDeconvolutionController:
    def __init__(self, model):
        self.model = model
        self.original_signal = None
        self.observed_signal = None
        self.reconstructed = None

    def run_simulation(self, noise_level, prior_strength):
        """运行模拟并保存结果"""
        results = self.model.run_simulation(noise_level, prior_strength)
        self.original_signal, self.observed_signal, self.reconstructed = results

    def get_results(self):
        """获取最新的模拟结果"""
        return self.original_signal, self.observed_signal, self.reconstructed


# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#
#     # 初始化模型
#     model = BayesianDeconvolutionModel()
#
#     # 初始化控制器
#     controller = BayesianDeconvolutionController(model)
#
#     # 初始化视图
#     view = BayesianDeconvolutionView(controller)
#
#     # 初始模拟并更新视图
#     # 使用默认参数运行一次模拟
#     controller.run_simulation(0.2, 0.05)
#     original, observed, reconstructed = controller.get_results()
#     view.update_plots(original, observed, reconstructed)
#
#     # 显示窗口并运行应用
#     view.show()
#     sys.exit(app.exec_())