#%%
from keras import backend as K
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# Grad-CAMに必要な関数
def get_img_array(img, size):
    # `img` を size にリサイズし、arrayとして返します。
    img = tf.image.resize(img, size)
    img = np.expand_dims(img, axis=0)
    return img

# MNISTデータセットをロードし、前処理
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# データにチャンネル次元を追加
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), padding = "same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    # layers.Conv2D(64, (3, 3), activation='relu', padding="same"),
    # layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# テスト画像を取得
# output = model(tf.keras.Input(tf.convert_to_tensor(img_array)))
# model.input

# res = model.predict(x_test[0:1])

# 最後の畳み込み層と出力層の名前を指定


#%%
img = x_test[23]
img_array = get_img_array(img, size=(28, 28))

last_conv_layer_name = "conv2d_2"
classifier_layer_names = [
    "flatten",
    "dense",
    "dense_1",
]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    # グラデーションに基づくクラス活性化マップを生成します。

    grad_model = tf.keras.models.Model(
        [model.inputs[0]],
        [model.get_layer(last_conv_layer_name).output, model.outputs]
    )

    # a = grad_model(img_array)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        # print(conv_outputs)
        # print(np.argmax(predictions[0][0]))
        loss = predictions[0][0][int(np.argmax(predictions[0][0]))]

    # print(conv_outputs)
    grads = tape.gradient(loss, conv_outputs)
    # print(grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    # print(heatmap)
    return heatmap.numpy()
# Grad-CAM heatmapを作成
# heatmap = get_img_array(heatmap, (28, 28))
# print(heatmap)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def overlay_heatmap(img: np.ndarray, heatmap: np.ndarray):
    # imgとheatmapのサイズを取得
    img_size = img.shape[0]
    heatmap_size = heatmap.shape[0]
    # heatmapをimgのサイズにリサイズ
    zoom_factor = img_size / heatmap_size
    resized_heatmap = zoom(heatmap, zoom_factor)
    # 画像を表示
    plt.imshow(img, cmap='gray', interpolation='nearest')
    # heatmapを重ねる
    plt.imshow(resized_heatmap, cmap='jet', alpha=0.5, interpolation='bilinear')
    # 表示
    plt.colorbar()  # ヒートマップのカラーバーを表示
    plt.show()

def display_gradcam(img, heatmap, alpha=0.4):
    # Grad-CAMを表示するための関数
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    jet = plt.cm.get_cmap("jet")


    jet_heatmap = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_heatmap[heatmap]


    superimposed_img = jet_heatmap * alpha + img
    # superimposed_img = img

    superimposed_img = tf.image.resize(superimposed_img, (28, 28))

    plt.imshow(tf.keras.preprocessing.image.array_to_img(superimposed_img))
    plt.colorbar()
    plt.show()

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names)
overlay_heatmap(np.squeeze(img_array), heatmap)
# print(heatmap.shape)
# 元の画像にヒートマップを適用して表示
# plt.imshow(heatmap)
# display_gradcam(img, heatmap)


#%%
