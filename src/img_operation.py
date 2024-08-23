#%%
import numpy as np
from skimage.transform import resize
from file_operator.yaml_operation import read_yaml

def trim_image(img: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    if np.all(img == 0): # 0matrixはそのまま返す
        return img
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


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    if np.ndim(img) == 1:
        return resize(img, (size, ), anti_aliasing=True, preserve_range=True)

    elif np.ndim(img) == 2:
        # 画像をリサイズして、出力をfloatから元のdtypeに変換
        return resize(img, (size, size), anti_aliasing=True, preserve_range=True)
        # 出力を元のdtypeに変換して返す
        return resized_img.astype(img.dtype)
    else:
        raise ValueError("the dimension of img is 1 or 2")

if __name__=="__main__":

    def test_trim_image():
        path = "../test/persistent_img/file_0a26d31d584e01ff6d157e268603d0e5.npy"
        d = np.load(path)[0]
        threshold = 0.01
        res = trim_image(d, threshold=threshold)
        assert np.abs(np.max(d) * threshold - np.min(res)) < threshold # resにはmax(d)*0.01とそう遠くない数値があるはず
        assert d.shape[0] > res.shape[0]
        assert d.shape[1] > res.shape[1]

    def test_resize_image():
        path = "../test/persistent_img/file_0a26d31d584e01ff6d157e268603d0e5.npy"
        d = np.load(path)[0]
        matrix_size = 100
        resized_res = resize_image(d, matrix_size)
        assert resized_res.shape[0] == matrix_size
        assert resized_res.shape[1] == matrix_size

    def test_trim_matrix_0_expected_same_matrix():
        # 0 matrixはトリミングしない
        zero_mat = np.zeros((10,10))
        trimed_zero_mat = trim_image(zero_mat)
        assert np.all(zero_mat == trimed_zero_mat)

    import matplotlib.pyplot as plt
    def test_resize_vector():
        img_vector = np.linspace(0, 10, 100)
        resized_vector = resize_image(img_vector, 10)
        assert resized_vector.shape == (10,)

    test_resize_vector()
    test_trim_image()
    test_resize_image()
    test_trim_matrix_0_expected_same_matrix()

#%%