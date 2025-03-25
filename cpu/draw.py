import os
import numpy as np
import matplotlib.pyplot as plt
import sys

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
        dirs = self.filename.split('/')
        file = dirs[len(dirs) - 1][:-4]
        print(file)
        ylabel = "Score"
        title = file
        if(file.find("(log)") >= 0):
            ylabel = ylabel + "(log2)"
            title = title.replace("(log)", "chromosome")
        else:
            title = title[:-1] + "chromosome" + title[-1:]

        print("title = {}\n".format(title))
        print("ylabel = {}\n".format(ylabel))

        plt.plot(self.data, marker='o')
        plt.xlabel('Index')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.show()


def main():
    if len(sys.argv) < 2:
        print("請輸入檔案名")
        sys.exit(1)

    # 建立 MyDataPlotter 物件
    plotter = Drawer("../output/" + sys.argv[1])
    # 讀取資料
    plotter.load_data()
    # 繪圖
    plotter.plot_data()

if __name__ == '__main__':
    main()