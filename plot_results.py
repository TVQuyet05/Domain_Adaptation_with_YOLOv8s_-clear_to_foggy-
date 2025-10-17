import pandas as pd
import matplotlib.pyplot as plt

# Đọc file results.csv
df = pd.read_csv('results.csv')

# Loại bỏ cột epoch nếu YOLO đã tự thêm (nếu có thì giữ lại để dùng làm trục X)
x = df.index if 'epoch' not in df.columns else df['epoch']

# Vẽ tất cả các cột trừ epoch
for col in df.columns:
    if col != 'epoch':
        plt.figure()
        plt.plot(x, df[col])
        plt.title(col)
        plt.xlabel('Epoch')
        plt.ylabel(col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
