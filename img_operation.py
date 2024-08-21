import numpy as np
from file_operator.yaml_operation import read_yaml
import tda_for_phase_field.tda as tda

def trim_image(img: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    # しきい値に基づいてマスクを作成
    max_value = np.max(img)
    mask = img > max_value * threshold
    # 有効な領域の最小および最大のインデックスを取得
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    # トリミング範囲を正方形に調整
    row_center = (row_min + row_max) // 2
    col_center = (col_min + col_max) // 2
    side_length = max(row_max - row_min, col_max - col_min)
    row_min = max(0, row_center - side_length // 2)
    row_max = min(img.shape[0], row_min + side_length)
    col_min = max(0, col_center - side_length // 2)
    col_max = min(img.shape[1], col_min + side_length)
    # 正方形領域をトリミング
    trimmed_img = img[row_min:row_max, col_min:col_max]
    # 正方形サイズを保証するため、再度サイズ調整
    final_size = min(trimmed_img.shape)
    trimmed_img = trimmed_img[:final_size, :final_size]
    return trimmed_img
from skimage.transform import resize

def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    # 画像をリサイズして、出力をfloatから元のdtypeに変換
    resized_img = resize(img, (size, size), anti_aliasing=True, preserve_range=True)
    # 出力を元のdtypeに変換して返す
    return resized_img.astype(img.dtype)
#%%
if __name__=="__main__":
    datas = read_yaml("summary/used_in_NN.yaml")
    # x = list(map(lambda x: np.transpose(np.load(x["persistent_img_path"]), (1,2,0)), datas))

    data = datas[0]
    d = np.load(data["persistent_img_path_hom1"])

    res = trim_image(d[0])
    resized_res = resize_image(res, 100)
    tda.plot_persistence_image_from_tda_diagram(resized_res)
    res.shape
#%%