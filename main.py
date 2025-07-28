import sys

from PyQt5.QtWidgets import QApplication

from core.logic.deconv_processor import BayesianDeconvolutionModel, BayesianDeconvolutionController, \
    BayesianDeconvolutionView

def load_stylesheet():
    with open('ui/styles/main_style.qss', 'r', encoding='utf-8') as f:
        return f.read()

# def show_window():
#     window.show()
#     window.activateWindow()  # 激活窗口

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 初始化模型
    model = BayesianDeconvolutionModel()

    # 初始化控制器
    controller = BayesianDeconvolutionController(model)

    # 初始化视图
    view = BayesianDeconvolutionView(controller)

    # 初始模拟并更新视图
    # 使用默认参数运行一次模拟
    controller.run_simulation(0.2, 0.05)
    original, observed, reconstructed = controller.get_results()
    view.update_plots(original, observed, reconstructed)

    # 显示窗口并运行应用
    view.show()
    sys.exit(app.exec_())
