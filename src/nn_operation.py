from tensorflow.keras import layers, models, Input, regularizers
from tensorflow.keras.datasets import mnist
import numpy as np
from sklearn.model_selection import train_test_split


def make_mini_lnn_block(input_x):
    input = Input(shape=input_x[0].shape)
    block = layers.Dense(
        8, activation="relu", kernel_regularizer=regularizers.l2(0.005)
    )(input)
    return input, block


def make_lnn_block(input_x):
    input = Input(shape=input_x[0].shape)
    # a1 = layers.Flatten()(input)
    block = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.005))(input)
    # block = layers.Dense(32, activation="relu")(block)
    # block = layers.Dense( 16, activation="relu", kernel_regularizer=regularizers.l2(0.01))(block)
    # block = layers.Dense(10, activation='relu')(block)
    block = layers.Flatten()(block)
    return input, block

def make_cnn_mini_block(input_x):
    input = Input(shape=input_x[0].shape)
    block = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(block)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.005),
    )(block)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Flatten()(block)
    return input, block


def make_cnn_block(input_x):
    input = Input(shape=input_x[0].shape)
    block = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(block)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.005),
    )(block)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Conv2D(
        256,
        (2, 2),
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(0.005),
    )(block)
    block = layers.MaxPooling2D((2, 2))(block)
    block = layers.Flatten()(block)
    return input, block


def make_lnn_combined_block(block_list, output_y):
    combined = layers.concatenate(block_list)
    # 全結合層を通じて最終的な出力を生成
    # z = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01))(combined)
    # z = layers.Dense(128, activation="relu")(combined)
    # z = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(combined)
    z = layers.Dense(16, activation="relu", kernel_regularizer=regularizers.l2(0.005))(
        combined
    )
    z = layers.Dense(len(output_y[0]))(z)
    return z


def pick_up_datas(func, datas):
    # def wrapper(func):
    #     return [func]
    return np.array(list(map(func, datas)))


def divide_data_as_test_valid_train(x, y, test_size=0.4, valid_size=0.5):
    x_train, _x_test, y_train, _y_test = train_test_split(
        x, y, test_size=test_size, random_state=123
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        _x_test, _y_test, test_size=valid_size, random_state=1234
    )
    return (x_train, y_train), (x_validation, y_validation), (x_test, y_test)


def apply_nn_to_x(x_unzip_list, nn_block_fn_list):
    if not len(x_unzip_list) == len(nn_block_fn_list):
        raise ValueError("x_unzip_list and nn_block_fn_list should same length")

    n = len(nn_block_fn_list)
    input_list = []
    block_list = []

    for i in range(n):
        nn_fn = nn_block_fn_list[i]
        input, block = nn_fn(x_unzip_list[i])
        input_list.append(input)
        block_list.append(block)

    return input_list, block_list


def unzip(zipped_x):
    lx = list(zip(*zipped_x))
    res = []
    for l in lx:
        res.append(np.array(l))
    return res


# x0 = np.array(
#     list(map(lambda x: [x["information"]["c10"], x["information"]["c20"]], datas))
# )
# x0 = list(map(lambda x: np.load(x["persistent_img_path_hom0"]), datas))
# x1 = np.array(list(map(lambda x: imop.resize_image(np.load(x["persistent_img_path_hom0"]), 38), datas)))


# x2 = np.array(list(map(lambda x: get_img(x), datas)))
# x2 = np.array(list(map(lambda x: get_img_for_binary(x), datas)))
# x2 = np.expand_dims(x2, axis=-1)
# d2 = map(lambda x: np.load(x["persistent_img_path_hom1"]), datas)
# d2 = map(lambda x: imop.resize_image(imop.trim_image(x, 0.01), 32), d2)

# x1 = np.array(x1)
# x2 = np.array(x2)
# k = np.array()
# k.shape

# x = list(map(lambda x: np.array(
#     np.transpose(
#         [
#             np.load(x["file_list"][-1][0]),
#             np.load(x["file_list"][-1][1])
#         ], (1,2,0))), datas))
# y = list(map(lambda x: [x["information"]["k11"], x["information"]["k12"], x["information"]["k22"]], datas))
# y = list(
#     map(
#         lambda x: [
#             x["information"]["w12"],
#             x["information"]["w23"],
#             x["information"]["w13"],
#         ],
#         datas,
#     )
# )
# y = list(map(lambda x: [x["information"]["c10"], x["information"]["c20"]], datas))
