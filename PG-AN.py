import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K

# parameters
COUNTY = 10335
BATCH_SIZE = COUNTY // 10  # county number
EPOCH_TRAIN = 10
LEARNING_RATE = 0.01
MODE = 'regression'
REGRESSION_LOSS = 'mean_squared_error'

# Synthetic Data
synthetic_X = np.load('synthetic_X.npy')
synthetic_ra = np.load('synthetic_Ra.npy')
synthetic_rh = np.load('synthetic_Rh.npy')
synthetic_nee = np.load('synthetic_NEE.npy')
synthetic_yield = np.load('synthetic_y.npy')
print(synthetic_X.shape, synthetic_ra.shape, synthetic_rh.shape, synthetic_nee.shape)

X_scaler = np.load('synthetic_X_scaler.npy')
Y1_scaler = np.load('synthetic_y1_scaler.npy')
Y2_scaler = np.load('synthetic_y2_scaler.npy')


def Z_norm_reverse(X, Xscaler, units_convert=1.0):
    return (X * Xscaler[1] + Xscaler[0]) * units_convert


def Z_norm_with_scaler(X, Xscaler):
    return (X - Xscaler[0]) / Xscaler[1]


synthetic_X = np.reshape(synthetic_X, (18, 365, 10335, 19))
synthetic_ra = np.reshape(synthetic_ra, (18, 365, 10335))
synthetic_rh = np.reshape(synthetic_rh, (18, 365, 10335))
synthetic_nee = np.reshape(synthetic_nee, (18, 365, 10335))
print(synthetic_X.shape, synthetic_ra.shape, synthetic_rh.shape, synthetic_nee.shape, synthetic_yield.shape)
synthetic_X = np.swapaxes(synthetic_X, 1, 2)
synthetic_ra = np.swapaxes(synthetic_ra, 1, 2)
synthetic_rh = np.swapaxes(synthetic_rh, 1, 2)
synthetic_nee = np.swapaxes(synthetic_nee, 1, 2)
print(synthetic_X.shape, synthetic_ra.shape, synthetic_rh.shape, synthetic_nee.shape, synthetic_yield.shape)

synthetic_X = np.reshape(synthetic_X, (COUNTY * 18, 365, 19))
synthetic_ra = np.reshape(synthetic_ra, (COUNTY * 18, 365))
synthetic_rh = np.reshape(synthetic_rh, (COUNTY * 18, 365))
synthetic_nee = np.reshape(synthetic_nee, (COUNTY * 18, 365))
synthetic_yield = np.reshape(synthetic_yield, (COUNTY * 18, 1))
print(synthetic_X.shape, synthetic_ra.shape, synthetic_rh.shape, synthetic_nee.shape, synthetic_yield.shape)

# Split synthetic data into train & test
synthetic_X_train = synthetic_X[::2]
synthetic_gpp_train = synthetic_X[::2, :, 8]
synthetic_ra_train = synthetic_ra[::2]
synthetic_rh_train = synthetic_rh[::2]
synthetic_nee_train = synthetic_nee[::2]
synthetic_yield_train = synthetic_yield[::2]
print(synthetic_X_train.shape, synthetic_gpp_train.shape, synthetic_ra_train.shape, synthetic_rh_train.shape,
      synthetic_nee_train.shape, synthetic_yield_train.shape)

synthetic_X_test = synthetic_X[1::2]
synthetic_gpp_test = synthetic_X[1::2, :, 8]
synthetic_ra_test = synthetic_ra[1::2]
synthetic_rh_test = synthetic_rh[1::2]
synthetic_nee_test = synthetic_nee[1::2]
synthetic_yield_test = synthetic_yield[1::2]
print(synthetic_X_test.shape, synthetic_gpp_test.shape, synthetic_ra_test.shape, synthetic_rh_test.shape,
      synthetic_nee_test.shape, synthetic_yield_test.shape)

# Yield data; index 8: gpp, index 9 : year
raw_X = np.load('Corn_X_set_two_states.npy')
raw_y = np.load('Corn_y_set_two_states.npy')

# First 19 years for training
TRAIN_MASK = np.zeros(21).astype(int)
TRAIN_MASK[:19] = True
print(TRAIN_MASK)

# compare curve with y data and normalized y data
# plt.plot(raw_y[1, :])
# plt.show()
# raw_y = Z_norm_with_scaler(raw_y, Y2_scaler[0, :])
# plt.plot(raw_y[1, :])
# plt.show()

print('Size of feature raw set: ', raw_X.shape)
print('Size of label raw set: ', raw_y.shape)

X = np.reshape(raw_X, (199, 21, 365, 19))

print('Size of feature set: ', X.shape)
print('Size of label set: ', raw_y.shape)

yield_X_train = X[:, TRAIN_MASK == 1, :, :]
yield_y_train = raw_y[:, TRAIN_MASK == 1]

yield_X_train = np.swapaxes(yield_X_train, 0, 1)
print('Size of feature training set: ', yield_X_train.shape)
yield_X_train = np.reshape(yield_X_train, (199 * 19, 365, 19))
yield_y_train = np.swapaxes(yield_y_train, 0, 1)
yield_y_train = np.reshape(yield_y_train, -1)

yield_gpp_train = yield_X_train[:, :, 8]
print('Size of feature training set: ', yield_X_train.shape)
print('Size of label training set: ', yield_y_train.shape)
print('Size of gpp training set: ', yield_gpp_train.shape)

yield_X_test = X[:, TRAIN_MASK == 0, :, :]
yield_y_test = raw_y[:, TRAIN_MASK == 0]

yield_X_test = np.swapaxes(yield_X_test, 0, 1)
print('Size of feature training set: ', yield_X_test.shape)
yield_X_test = np.reshape(yield_X_test, (199 * 2, 365, 19))
yield_y_test = np.swapaxes(yield_y_test, 0, 1)
yield_y_test = np.reshape(yield_y_test, -1)

yield_gpp_test = yield_X_test[:, :, 8]
print('Size of feature testing set: ', yield_X_test.shape)
print('Size of label testing set: ', yield_y_test.shape)
print('Size of gpp testing set: ', yield_gpp_test.shape)

# for model input, dimensions should be same, repeat yield data. also can use np.repeat()
idx = np.random.permutation(range(yield_X_train.shape[0]))
yield_X_train, yield_gpp_train, yield_y_train = yield_X_train[idx], yield_gpp_train[idx], yield_y_train[idx]
print(yield_X_train.shape, yield_gpp_train.shape, yield_y_train.shape)
for i in range(synthetic_X_train.shape[0] // yield_X_train.shape[0] - 1):
    if i == 0:
        combined_yield_X_train = np.concatenate((yield_X_train, yield_X_train), axis=0)
        combined_yield_gpp_train = np.concatenate((yield_gpp_train, yield_gpp_train), axis=0)
        combined_yield_y_train = np.concatenate((yield_y_train, yield_y_train), axis=0)
    else:
        combined_yield_X_train = np.concatenate((combined_yield_X_train, yield_X_train), axis=0)
        combined_yield_gpp_train = np.concatenate((combined_yield_gpp_train, yield_gpp_train), axis=0)
        combined_yield_y_train = np.concatenate((combined_yield_y_train, yield_y_train), axis=0)
remain_idx = synthetic_X_train.shape[0] % yield_X_train.shape[0]
combined_yield_X_train = np.concatenate((combined_yield_X_train, yield_X_train[:remain_idx]), axis=0)
combined_yield_gpp_train = np.concatenate((combined_yield_gpp_train, yield_gpp_train[:remain_idx]), axis=0)
combined_yield_y_train = np.concatenate((combined_yield_y_train, yield_y_train[:remain_idx]), axis=0)
print(combined_yield_X_train.shape, combined_yield_gpp_train.shape, combined_yield_y_train.shape)

# for test data
idx = np.random.permutation(range(yield_X_test.shape[0]))
yield_X_test, yield_gpp_test, yield_y_test = yield_X_test[idx], yield_gpp_test[idx], yield_y_test[idx]
print(yield_X_test.shape, yield_gpp_test.shape, yield_y_test.shape)
for i in range(synthetic_X_train.shape[0] // yield_X_test.shape[0] - 1):
    if i == 0:
        combined_yield_X_test = np.concatenate((yield_X_test, yield_X_test), axis=0)
        combined_yield_gpp_test = np.concatenate((yield_gpp_test, yield_gpp_test), axis=0)
        combined_yield_y_test = np.concatenate((yield_y_test, yield_y_test), axis=0)
    else:
        combined_yield_X_test = np.concatenate((combined_yield_X_test, yield_X_test), axis=0)
        combined_yield_gpp_test = np.concatenate((combined_yield_gpp_test, yield_gpp_test), axis=0)
        combined_yield_y_test = np.concatenate((combined_yield_y_test, yield_y_test), axis=0)
remain_idx = synthetic_X_train.shape[0] % yield_X_test.shape[0]
combined_yield_X_test = np.concatenate((combined_yield_X_test, yield_X_test[:remain_idx]), axis=0)
combined_yield_gpp_test = np.concatenate((combined_yield_gpp_test, yield_gpp_test[:remain_idx]), axis=0)
combined_yield_y_test = np.concatenate((combined_yield_y_test, yield_y_test[:remain_idx]), axis=0)
print(combined_yield_X_test.shape, combined_yield_gpp_test.shape, combined_yield_y_test.shape)

# training data for physical information
X_train = (
    synthetic_X_train, synthetic_gpp_train, combined_yield_X_train, combined_yield_gpp_train, combined_yield_X_test,
    combined_yield_gpp_test)
y_train = (synthetic_ra_train, synthetic_rh_train, synthetic_nee_train, synthetic_yield_train, combined_yield_y_train)

# model structure
initializer = tf.keras.initializers.TruncatedNormal(stddev=0.3)  # mean=0.0, seed=None

synthetic_X_input = tf.keras.layers.Input(shape=(365, 19))
synthetic_gpp = tf.keras.layers.Input(shape=(365,))
yield_train_X_input = tf.keras.layers.Input(shape=(365, 19))
yield_train_gpp = tf.keras.layers.Input(shape=(365,))
yield_test_X_input = tf.keras.layers.Input(shape=(365, 19))
yield_test_gpp = tf.keras.layers.Input(shape=(365,))

merged_input = tf.keras.layers.Concatenate(axis=0)([synthetic_X_input, yield_train_X_input, yield_test_X_input])

base_lstm = tf.keras.layers.GRU(64, activation='tanh', kernel_initializer=initializer, return_sequences=True)(
    merged_input)
base_drop = tf.keras.layers.Dropout(0.2)(base_lstm)
out_ra = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 64)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, kernel_initializer=initializer)]
)(base_lstm)
out_rh = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 64)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, kernel_initializer=initializer)]
)(base_lstm)
out_nee = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 64)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, kernel_initializer=initializer)]
)(base_lstm)
attn_yield = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 19)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=initializer)]
)(merged_input)
attn_yield = tf.nn.softmax(attn_yield, axis=1)
attn_yield = tf.reshape(attn_yield, (-1, 1, merged_input.shape[1]))
layer_yield = tf.squeeze(K.batch_dot(base_drop, attn_yield, axes=[1, 2]))
out_yield = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(64,)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, kernel_initializer=initializer)]
)(layer_yield)

synthetic_out_ra, yield_train_out_ra, yield_test_out_ra = tf.split(out_ra, num_or_size_splits=3, axis=0)
synthetic_out_rh, yield_train_out_rh, yield_test_out_rh = tf.split(out_rh, num_or_size_splits=3, axis=0)
synthetic_out_nee, yield_train_out_nee, yield_test_out_nee = tf.split(out_nee, num_or_size_splits=3, axis=0)
synthetic_out_yield, yield_train_out_yield, yield_test_out_yield = tf.split(out_yield, num_or_size_splits=3, axis=0)

temp_synthetic_gpp = Z_norm_reverse(synthetic_gpp, X_scaler[8, :], 1.0)
temp_synthetic_ra = Z_norm_reverse(synthetic_out_ra, Y1_scaler[0, :], 1.0)
temp_synthetic_rh = Z_norm_reverse(synthetic_out_rh, Y1_scaler[1, :], 1.0)
temp_synthetic_nee = Z_norm_reverse(synthetic_out_nee, Y1_scaler[2, :], 1.0)
synthetic_loss = 0.1 * K.mean(K.square(
    temp_synthetic_gpp + temp_synthetic_ra[:, :, 0] + temp_synthetic_rh[:, :, 0] + temp_synthetic_nee[:, :, 0]))

temp_yield_train_gpp = Z_norm_reverse(yield_train_gpp, X_scaler[8, :], 1.0)
temp_yield_train_ra = Z_norm_reverse(yield_train_out_ra, Y1_scaler[0, :], 1.0)
temp_yield_train_rh = Z_norm_reverse(yield_train_out_rh, Y1_scaler[1, :], 1.0)
temp_yield_train_nee = Z_norm_reverse(yield_train_out_nee, Y1_scaler[2, :], 1.0)
yield_train_loss = 0.1 * K.mean(K.square(
    temp_yield_train_gpp + temp_yield_train_ra[:, :, 0] + temp_yield_train_rh[:, :, 0] + temp_yield_train_nee[:, :, 0]))

temp_yield_test_gpp = Z_norm_reverse(yield_test_gpp, X_scaler[8, :], 1.0)
temp_yield_test_ra = Z_norm_reverse(yield_test_out_ra, Y1_scaler[0, :], 1.0)
temp_yield_test_rh = Z_norm_reverse(yield_test_out_rh, Y1_scaler[1, :], 1.0)
temp_yield_test_nee = Z_norm_reverse(yield_test_out_nee, Y1_scaler[2, :], 1.0)
yield_test_loss = 0.1 * K.mean(K.square(
    temp_yield_test_gpp + temp_yield_test_ra[:, :, 0] + temp_yield_test_rh[:, :, 0] + temp_yield_test_nee[:, :, 0]))

model_all = tf.keras.models.Model(
    inputs=[synthetic_X_input, synthetic_gpp, yield_train_X_input, yield_train_gpp, yield_test_X_input, yield_test_gpp],
    outputs=[synthetic_out_ra, synthetic_out_rh, synthetic_out_nee, synthetic_out_yield, yield_train_out_yield])

# Energy loss
model_all.add_loss(synthetic_loss)
model_all.add_loss(yield_train_loss)
model_all.add_loss(yield_test_loss)
model_all.add_metric(synthetic_loss, name='synthetic_conservation_loss', aggregation='mean')
model_all.add_metric(yield_train_loss, name='yield_train_conservation_loss', aggregation='mean')
model_all.add_metric(yield_test_loss, name='yield_test_conservation_loss', aggregation='mean')


def custom_loss(y_true_in, y_pred_in):
    mask = tf.math.is_nan(y_true_in)
    return tf.keras.losses.MeanSquaredError()(y_true_in[~mask], y_pred_in[~mask])


optimizer_all = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
# Each loss corresponds to each output
model_all.compile(optimizer=optimizer_all,
                  loss=[REGRESSION_LOSS, REGRESSION_LOSS, REGRESSION_LOSS, REGRESSION_LOSS, custom_loss])
model_all.summary()

# model train
# model_all.fit(X_train, y_train, batch_size=BATCH_SIZE, initial_epoch=0, epochs=EPOCH_TRAIN, shuffle=True)
model_all.load_weights("phy_model")

# Evaluation
# X_test = (combined_yield_X_test, combined_yield_gpp_test, combined_yield_X_train, combined_yield_gpp_train,
#           combined_yield_X_test,
#           combined_yield_gpp_test)
# # energy loss for prediction
# y_pred = model_all.predict(X_test, batch_size=BATCH_SIZE)
# pred_ra = Z_norm_reverse(y_pred[0][:, :, 0], Y1_scaler[0, :], 1.0)
# pred_rh = Z_norm_reverse(y_pred[1][:, :, 0], Y1_scaler[1, :], 1.0)
# pred_nee = Z_norm_reverse(y_pred[2][:, :, 0], Y1_scaler[2, :], 1.0)
# pred_gpp = -(pred_ra + pred_rh + pred_nee)
# real_gpp = Z_norm_reverse(X_test[1], X_scaler[8, :], 1.0)
# print(np.mean(np.square(pred_ra + pred_rh + pred_nee + real_gpp)))


# PG-AN
initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)  # mean=0.0, seed=None

input_X = tf.keras.layers.Input(shape=(365, 19))

base_lstm = tf.keras.layers.GRU(64, activation='tanh', kernel_initializer=initializer, return_sequences=True)(input_X)
base_drop = tf.keras.layers.Dropout(0.2)(base_lstm)
attn_yield = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 19)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=initializer)]
)(input_X)
attn_yield = tf.nn.softmax(attn_yield, axis=1)
attn_yield = tf.reshape(attn_yield, (-1, 1, input_X.shape[1]))
layer_yield = tf.squeeze(K.batch_dot(base_drop, attn_yield, axes=[1, 2]))
out_yield = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(64,)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
     tf.keras.layers.Dense(1, kernel_initializer=initializer)]  #
)(layer_yield)

model_X = tf.keras.models.Model(inputs=input_X, outputs=out_yield)


def custom_loss(y_true_in, y_pred_in):
    mask = tf.math.is_nan(y_true_in)
    return tf.keras.losses.MeanSquaredError()(y_true_in[~mask], y_pred_in[~mask])


optimizer_X = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
model_X.compile(optimizer=optimizer_X, loss=custom_loss)
model_X.summary()
model_X.layers[1].set_weights(model_all.layers[4].get_weights())
model_X.layers[2].set_weights(model_all.layers[5].get_weights())

# Training
model_X.fit(yield_X_train, yield_y_train, batch_size=BATCH_SIZE, initial_epoch=0, epochs=30, shuffle=True)


# Evaluation
def rmse_value(y_true_in, y_pred_in):
    return np.sqrt(np.nanmean(np.square(np.subtract(y_true_in, y_pred_in))))


y_pred_test = model_X.predict(yield_X_test, batch_size=BATCH_SIZE).squeeze()
print(rmse_value(yield_y_test, y_pred_test))
