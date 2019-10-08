import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.models import Model  # 泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import pandas as pd


def demo():
    # 数据预处理
    train_dataframe = pd.read_csv('./data/iris.data', header=0)
    # print(train_dataframe)
    train_dataset = train_dataframe.values
    total_count = 150
    train_level = int(total_count*0.7)

    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    x_test = train_dataset[train_level:, 0:-1].astype('float')
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    print(x_train.shape)
    print(x_test.shape)

    # 压缩特征维度至2维
    encoding_dim = 2

    # this is our input placeholder
    input_img = Input(shape=(4,))
    decoder_input = Input(shape=(encoding_dim,))

    # 编码层
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # 解码层
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(4, activation='tanh')(decoded)

    # 构建自编码模型
    autoencoder = Model(inputs=input_img, outputs=decoded)

    # 构建编码模型
    encoder = Model(inputs=input_img, outputs=encoder_output)

    # 构建解码模型
    # decoder = Model(inputs=decoder_input, outputs=decoded)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, epochs=20, batch_size=256, shuffle=True)

    # plotting
    encoded_imgs = encoder.predict(x_test)
    print(encoded_imgs)

    # decoded_imgs = decoder.predict(encoded_imgs)
    # print(decoded_imgs)

    autoencoder_img = autoencoder.predict(x_test)
    print(autoencoder_img)


if __name__ == '__main__':
    demo()