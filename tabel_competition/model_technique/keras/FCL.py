# required library
import matplotlib.pyplot as plt
import numyp as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from keras import losses
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Dropout, Input, Multiply, Add
from keras.layers.convolutional import Conv1D, UpSampling1D
from keras.layers.pooling import MaxPool1D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Adamax
from keras.optimizers import Adagrad
from keras.optimizers import Nadam
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import losses, models, optimizers

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

optimizer = Adagrad(lr=0.01)
optimizer = Adadelta(lr=1.0)
optimizer = Adamax(lr=0.002)
optimizer = Adam(lr=0.001)


def step_decay(epoch):
    if epoch < 50:
        return 0.001
    else:
        return 0.0005 * float(tf.math.exp((1 - epoch)))


# main wave_block
lr_decay = LearningRateScheduler(step_decay)


def create_mlp(shape):
    '''
    Returns a keras model
    '''
    print(f"shape: {shape}")
    model = Sequential()
    model.add(Dense(100, input_shape=shape, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(50, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
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

    mlp = create_mlp(X_train.values[0].shape)
    mlp.compile(optimizer=optimizer, loss=losses.mean_squared_error)
    mlp.fit(x=X_train.values, y=y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_valid.values, y_valid), callbacks=[lr_decay])
    mlp_pred = mlp.predict(X_valid.values)

    plt.plot(mlp.history.history['loss'][3:], 'r', label='loss', alpha=0.7)
    plt.plot(mlp.history.history['val_loss'][3:], label='val_loss', alpha=0.7)
    plt.show()

    mlp_pred = mlp.predict(np.reshape(X_valid.values, (-1, X_train.shape[1], 1)))
    score = np.sqrt(mean_squared_log_error(np.exp(y_valid), np.exp(mlp_pred)))
    print(score)
