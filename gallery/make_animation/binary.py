# %%
# 2dの相分離アニメーションを作成する
#===============================================
# Step 1: 計算
#===============================================
import numpy as np
import os
from phase_field_2d import PhaseField
from datetime import datetime
from matplotlib.colors import Normalize

phase_field = PhaseField(
    3.0,
    0.5,
    method="anisotropic",
    eta=np.array([1, 1, 0]),
    save_dir_name="../data/raw_data/c0_05_w_3_eta22_1",
    record=True,
)
phase_field.nprint = 10000
phase_field.nsave = 200
phase_field.nstep = 10000
phase_field.dtime = 0.005
phase_field.start()

# %%

#===============================================
# Step 2: 描画
#===============================================
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def sort_by_number(file_list):
    sorted_file_list = sorted(
        file_list, key=lambda x: int(x.split("_")[1].split(".")[0])
    )
    return sorted_file_list


# ディレクトリの中を走査
dir_path = "../data/c0_05_w_3_eta22_1/output_2024-09-05-12-30-29"
dir_list = os.listdir(dir_path)
# ディレクトリリストのうち数字を含むもの
dir_list = [f for f in dir_list if re.search(r'\d+', f)]
# 数字部分でソートする
dir_list = sort_by_number(dir_list)

images = []
for d in dir_list:
    d = np.load(f"{dir_path}/{d}")

    fig, ax = plt.subplots()
    ax.imshow(d, cmap='viridis')

    # 一時的なファイルに保存 (in-memory で保持する)
    fig.canvas.draw()  # Figure を描画
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # PillowのImageオブジェクトに変換してリストに追加
    images.append(Image.fromarray(image))

    plt.close(fig)  # 毎回閉じる

images[0].save('animated_image.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

# %%
