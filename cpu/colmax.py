import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

class Colmax :
    def __init__(self, filename):
        self.filename = filename
        self.data = []

    def load_data(self):
        with open(self.filename, "r") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    self.data.append(int(parts[2]))

    def plot_data(self):

        path = self.filename.split("/")
        plt.xlabel("Position")
        plt.ylabel("Score")
        plt.title("Chromosome" + path[len(path) -1][:-4])
        plt.plot(self.data)
        plt.show()

def main():
    if len(sys.argv) < 2:
        print("使用方法 <colmax.py>")
        sys.exit(1)

    # 建立 MyDataPlotter 物件
    plotter = Colmax("../output/" + sys.argv[1])
    # 讀取資料
    print("Loading data")
    plotter.load_data()

    print("Ploting")
    plotter.plot_data()

if __name__ == '__main__':
    main()
