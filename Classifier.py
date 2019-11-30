import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Model
from keras.layers import Dense, Input
import pandas as pd
import keras
from keras.utils import to_categorical


def test():
    # 数据预处理
    train_dataframe = pd.read_csv('./data/hu/train.txt', header=0)
    test_dataframe = pd.read_csv('./data/hu/train.txt', header=0)
    train_dataset = train_dataframe.values
    test_dataset = test_dataframe.values
    total_count = 980
    train_level = int(total_count * 0.7)

    # 训练集
    x_train = train_dataset[0:train_level, 0:-1].astype('float')
    y_train = train_dataset[0:train_level, -1].astype('int')

    # 验证集
    x_valid = train_dataset[train_level:, 0:-1].astype('float')
    y_valid = train_dataset[train_level:, -1].astype('int')

    # 测试集
    x_test = test_dataset[:, 0:-1].astype('float')
    y_test = test_dataset[:, -1].astype('int')

    # 数据转换
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    y_train = to_categorical(y_train)
    x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))
    y_valid = to_categorical(y_valid)
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    y_test = to_categorical(y_test)
    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    # 参数设置
    input_dim = 30  # 输入维度
    output_dim = 7  # 类别数目
    batch_size = 128  # 批数据
    epochs = 12  # 迭代次数
    # this is our input placeholder
    data_input = Input(shape=(input_dim,), name="data_input")

    # 编码层
    dense_1 = Dense(128, activation='relu', name='dense_1')(data_input)
    dense_2 = Dense(64, activation='relu', name='dense_2')(dense_1)
    dense_3 = Dense(10, activation='relu', name='dense_3')(dense_2)
    dense_4 = Dense(output_dim, name='dense_4')(dense_3)

    # 构建模型
    model = Model(inputs=data_input, outputs=dense_4)
    model.summary()
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_valid, y_valid))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    test()