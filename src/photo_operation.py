#%%
from PIL import Image
import numpy as np
from numpy.typing import NDArray

def image_to_grayscale_array_pil(image_path):
    # 画像を開く
    img = Image.open(image_path)
    # グレースケールに変換
    img_gray = img.convert('L')
    # NumPy配列に変換
    img_array = np.array(img_gray)
    return img_array

def convert_to_binary_img(img: NDArray):
    threshold = np.mean(img)
    return np.where(img >= threshold, 1, 0)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    img = image_to_grayscale_array_pil("data/rect11305.png")
    bimg = convert_to_binary_img(img)


