import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

class Segment:
        def __init__(self, start):
            self.start = start
            self.end = None
            self.maxV = None

class Drawer:

    def __init__(self, filename, stride, threshold):
        """初始化，指定要讀取的檔案名稱"""
        self.filename = filename
        self.data = []
        self.segments = []
        self.stride = int(stride)
        self.threshold = int(threshold)

    def load_data(self):
        with open(self.filename, "r") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    i, j, value = map(int, parts)
                    if(value > self.threshold):
                        self.data.append((int(j), int(i), int(value)))
        print(len(self.data))

    def make_segment(self):
        curSeg = Segment(0)
        maxV = self.data[0][2]
        for idx in tqdm(range(1, len(self.data))):
            if self.data[idx][0] - 1 == self.data[idx - 1][0]:
                diff = self.data[idx][2] - self.data[idx - 1][2]
                if self.data[idx][1] >= self.data[idx - 1][1] and (self.data[idx][1] - self.data[idx - 1][1] < self.stride)  and (((diff - 2) % 2  == 0) or ((diff + 3) % 2 == 0) or (diff % 2 == 0)):
                    maxV = max(maxV, self.data[idx][2])
                elif self.data[idx][1] == self.data[idx - 1][1]:
                    maxV = max(maxV, self.data[idx][2])
                else:
                    curSeg.end = idx
                    curSeg.maxV = maxV
                    self.segments.append(curSeg)
                    curSeg = Segment(idx)
            else:
                curSeg.end = idx
                curSeg.maxV = maxV
                self.segments.append(curSeg)
                curSeg = Segment(idx)

    def plot_data(self):
        """繪製折線圖"""
        for seg in tqdm(self.segments):
            if seg.maxV < self.threshold:
                continue
            points = self.data[seg.start: seg.end]
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            plt.plot(x, y)
            # plt.text(seg.end, y[len(y)-1], str(seg.maxV), fontsize=8)

        path = self.filename.split("/")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Chromosome" + path[len(path) -1][:-4])
        plt.show()
        

        


def main():
    if len(sys.argv) < 4:
        print("使用方法 <draw.py> <your output filename> <stride> <threshold>")
        sys.exit(1)

    # 建立 MyDataPlotter 物件
    plotter = Drawer("../output/" + sys.argv[1], sys.argv[2], sys.argv[3])
    # 讀取資料
    print("Loading data")
    plotter.load_data()

    print("Making segments")
    plotter.make_segment()

    print("Ploting")
    plotter.plot_data()

if __name__ == '__main__':
    main()