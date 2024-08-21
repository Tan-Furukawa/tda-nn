#%%
import os
from file_operator.base import save_str
import yaml
import matplotlib.pyplot as plt
from phase_field_2d_ternary.matrix_plot_tools import Ternary
#%%

def extract_directory_information(dir_path: str):
    result = []

    # ディレクトリの中を走査
    for subdir_name in os.listdir(dir_path):
        subdir_path = os.path.join(dir_path, subdir_name)

        if os.path.isdir(subdir_path):
            # 各サブディレクトリの情報を格納する辞書を作成
            dir_info = {
                "dir_name": os.path.abspath(subdir_path),  # 絶対パスに変更
                "file_list": [],
                "information": None
            }

            con1_files = []
            con2_files = []

            # サブディレクトリ内のファイルを走査
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)

                if file_name.startswith('con1') and file_name.endswith('.npy'):
                    con1_files.append(file_path)  # 絶対パスに変更
                elif file_name.startswith('con2') and file_name.endswith('.npy'):
                    con2_files.append(file_path)  # 絶対パスに変更
                elif file_name == "test.yaml":
                    with open(file_path, 'r') as yaml_file:
                        dir_info["information"] = yaml.safe_load(yaml_file)

            # con1とcon2のファイルをペアにする
            for con1_file in sorted(con1_files):
                matching_number = os.path.basename(con1_file).split('_')[1].split('.')[0]
                con2_file = next((f for f in con2_files if os.path.basename(f).split('_')[1].split('.')[0] == matching_number), None)
                if con2_file:
                    dir_info["file_list"].append([con1_file, con2_file])
            result.append(dir_info)
    return result

if __name__ == "__main__":
    import numpy as np
    dir_path = "result"
    directory_info = extract_directory_information(dir_path)
    res = yaml.dump(directory_info)
    save_str("summary/result_summary.yaml", res)


    for i, file in enumerate(directory_info):
        if i > 100:
            break
        f = file["file_list"]
        if len(f) == 1:
            continue
        c1 = np.load(f[-1][0])
        c2 = np.load(f[-1][1])
        Ternary.imshow3(c1, c2)
        plt.show()

    # %%
