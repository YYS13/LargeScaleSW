import numpy as np
import matplotlib.pyplot as plt

class Drawer:

    def __init__(self, filename):
        """初始化，指定要讀取的檔案名稱"""
        self.filename = filename
        self.data = None

    def load_data(self):
        """從檔案讀取資料 (一行一個整數)，並存到 self.data"""
        self.data = np.loadtxt(self.filename, dtype=float)

    def plot_data(self):
        """繪製折線圖"""
        if self.data is None:
            print("No data loaded. Please call load_data() first.")
            return

        plt.plot(self.data, marker='o')
        plt.xlabel('Index')
        plt.ylabel('Score(log2)')
        plt.title('Result')
        plt.grid(True)
        plt.show()


def main():
    # 建立 MyDataPlotter 物件
    plotter = Drawer("../output/result.txt")
    # 讀取資料
    plotter.load_data()
    # 繪圖
    plotter.plot_data()

if __name__ == '__main__':
    main()