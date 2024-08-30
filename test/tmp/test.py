#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# サンプルデータを生成
np.random.seed(0)
x = np.random.rand(100) * 10
y = 2.5 * x + np.random.randn(100) * 5

# データフレームに変換
data = pd.DataFrame({'x': x, 'y': y})

# Seabornの散布図に回帰線と信頼区間を追加
sns.regplot(x='x', y='y', data=data, ci=90)  # ciは信頼区間の範囲を指定します（デフォルトは95）

# プロットを表示
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.title('Scatter Plot with 95% Confidence Interval')
plt.show()

# %%
