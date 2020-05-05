# required library
import matplotlib.pyplot as plt
import numyp as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Multiply

# prepare data
train = pd.DataFrame()
test = pd.DataFrame()
X = train.drop("y", axis=1)
y = train["y"]

# parameters
CV_FOLD_NUM = 4
lerning_rate = 0.0001
epochs = 100
batch_size = 400


def step_decay(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.0005 * float(tf.math.exp((1 - epoch)))


# main wave_block
opt = Adam(lr=lerning_rate)
optimizer = tfa.optimizers.SWA(opt)
lr_decay = LearningRateScheduler(step_decay)


def cbr(x, out_layer, kernel, stride, dilation):
    x = Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2**i for i in range(n)]
    x = Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate)(x)
        sigm_out = Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate)(x)
        x = Multiply()([tanh_out, sigm_out])
        x = Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = Add()([res_x, x])
    return res_x


def create_mlp(shape):
    '''
    Returns a keras model
    '''
    print(f"shape: {shape}")
    inp = Input(shape=shape)
    x = cbr(inp, 64, 7, 1, 1)
    x = BatchNormalization()(x)
    x = wave_block(x, 16, 3, 12)
    x = BatchNormalization()(x)
    x = wave_block(x, 32, 3, 8)
    x = BatchNormalization()(x)
    x = wave_block(x, 64, 3, 4)
    x = BatchNormalization()(x)
    x = wave_block(x, 128, 3, 1)
    x = cbr(x, 32, 7, 1, 1)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    out = Dense(1, name='out')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model


kf = KFold(n_splits=CV_FOLD_NUM,
           shuffle=True,
           random_state=42)
scores = []
feature_importance_df = pd.DataFrame()

pred_cv = np.zeros(len(test.index))


for i, (tdx, vdx) in enumerate(kf.split(X, y)):
    print(f'Fold : {i}')
    X_train, X_valid, y_train, y_valid = X.iloc[tdx], X.iloc[vdx], y.values[tdx], y.values[vdx]

    mlp = create_mlp((X_train.values.shape[1], 1))
    mlp.compile(optimizer=optimizer, loss=losses.mean_squared_error)
    mlp.fit(x=np.reshape(X_train.values, (-1, X_train.shape[1], 1)), y=y_train.reshape(len(y_train), 1),
            epochs=epochs, batch_size=batch_size,
            validation_data=(np.reshape(X_valid.values, (-1, X_valid.shape[1], 1)), y_valid),
            callbacks=[lr_decay])

    plt.plot(mlp.history.history['loss'][3:], 'r', label='loss', alpha=0.7)
    plt.plot(mlp.history.history['val_loss'][3:], label='val_loss', alpha=0.7)
    plt.show()

    mlp_pred = mlp.predict(np.reshape(X_valid.values, (-1, X_train.shape[1], 1)))
    score = np.sqrt(mean_squared_log_error(np.exp(y_valid), np.exp(mlp_pred)))
    print(score)
