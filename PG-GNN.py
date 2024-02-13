import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
from libpysal.weights import Rook
import scipy.spatial as sp
from keras.callbacks import Callback

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

COUNTY = 199

BATCH_SIZE = COUNTY  # county number

INPUT_SIZE = 19
NUM_LAYERS_DNN = 10  # num_layers is the number of non-input layers, two more LSTM
NUM_CLASS = 1
TIME_STEP = 365

# data preprocessing
TRAIN_RATIO = 19
TEST_RATIO = 2

CKPT_FOLDER_PATH = 'share/'
EPOCH_TRAIN = 100
LEARNING_RATE = 0.001

REGRESSION_LOSS = 'mean_squared_error'
CLASSIFICATION_LOSS = 'binary_crossentropy'


# Models

class GraphMask(keras.layers.Layer):
    def __init__(self, init_m, state=True):
        super(GraphMask, self).__init__()
        self.w = tf.Variable(init_m, dtype="float32", trainable=state, name='graph_mask')

    def get_vars(self):
        return self.w

    def call(self, t_adj):
        out = tf.math.multiply(self.w, t_adj)
        return K.clip(out, 0, 1)


def global_model(init_graph, graph_train=False):
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.5)  # mean=0.0, seed=None

    input_x = tf.keras.layers.Input(shape=(365, 19))
    input_adj = tf.keras.layers.Input(shape=199)

    base_lstm = tf.keras.layers.GRU(64, activation='tanh', kernel_initializer=initializer, return_sequences=True)(
        input_x)
    base_drop = tf.keras.layers.Dropout(0.2)(base_lstm)

    attn_yield = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(365, 19)),
         tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
         tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
         tf.keras.layers.Dense(1, activation='tanh', kernel_initializer=initializer)]
    )(input_x)

    attn_yield = tf.nn.softmax(attn_yield, axis=1)
    attn_yield = tf.reshape(attn_yield, (-1, 1, input_x.shape[1]))
    layer_yield = tf.squeeze(K.batch_dot(base_drop, attn_yield, axes=[1, 2]))

    input_sim_output = GraphMask(init_graph, graph_train)(input_adj)

    # mean
    num_neighbours = K.sum(tf.squeeze(input_sim_output), axis=1)
    sum_aggregation = K.batch_dot(tf.reshape(input_sim_output, (-1, 199, 199)), tf.reshape(layer_yield, (-1, 199, 64)),
                                  axes=[2, 1])
    mean_aggregation = tf.math.divide_no_nan(sum_aggregation, num_neighbours[:, None])
    layer_yield_1 = tf.concat([layer_yield, tf.squeeze(mean_aggregation)], 1)
    layer_yield_2 = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(128,))])(
        layer_yield_1)

    out_yield = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer, input_shape=(64,)),
         tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
         tf.keras.layers.Dense(32, activation='relu', kernel_initializer=initializer),
         tf.keras.layers.Dense(1, kernel_initializer=initializer)]  #
    )(layer_yield_2)

    model = tf.keras.models.Model(inputs=[input_x, input_adj], outputs=out_yield)

    return model


def gnn_rmse_testing_evaluation(model_in, x_in, y_in, adj_in, ratio_in):
    y_pred_in = None
    for each_batch_in in range(ratio_in):
        current_y_pred_in = model_in.predict(
            [x_in[each_batch_in * 199:(each_batch_in + 1) * 199], adj_in], batch_size=BATCH_SIZE, verbose=0)
        current_y_pred_in = current_y_pred_in.squeeze()
        if y_pred_in is None:
            y_pred_in = current_y_pred_in.copy()
        else:
            y_pred_in = np.concatenate([y_pred_in, current_y_pred_in], axis=0)
    y_pred_in = np.reshape(y_pred_in, (-1, 1))
    y_true_in = np.reshape(y_in, (-1, 1))
    mask = np.isnan(y_true_in)
    rmse_test = np.sqrt(np.mean(K.square((y_true_in - y_pred_in))[~mask]))
    return rmse_test


def gnn_train(model_in, x_in, y_in, adj_in, epoch_train=EPOCH_TRAIN, test_data=None,
              ft=False):
    history_in = {'rmse_tr': [], 'rmse_test': []}
    random_batch_index_in = np.arange(19)
    x_in_conv = np.reshape(x_in, (19, COUNTY, 365, 19))
    y_in_conv = np.reshape(y_in, (19, -1))
    x_test_in, y_test_in = test_data

    for init_epoch_number_in in range(epoch_train):
        np.random.shuffle(random_batch_index_in)
        for each_batch_in in range(19):
            model_in.fit(
                [x_in_conv[random_batch_index_in[each_batch_in]], adj_in],
                y_in_conv[random_batch_index_in[each_batch_in]], batch_size=199, initial_epoch=init_epoch_number_in,
                epochs=init_epoch_number_in + 1,
                verbose=0, shuffle=False)

        # evaluation

        rmse_tr = gnn_rmse_testing_evaluation(model_in, x_in.copy(), y_in.copy(), adj_in.copy(),
                                               19)
        rmse_test = gnn_rmse_testing_evaluation(model_in, x_test_in.copy(), y_test_in.copy(), adj_in.copy(),
                                                2)

        print(K.eval(model_in.optimizer.lr))
        print(
            "epoch: {:2d}  evaluation - tr_rmse: {:.6f} - test_rmse: {:.6f}".format(init_epoch_number_in, rmse_tr,
                                                                                    rmse_test))

        history_in['rmse_tr'].append(rmse_tr)
        history_in['rmse_test'].append(rmse_test)

        if ft:
            temp_in = (init_epoch_number_in + 1)
            K.set_value(model_in.optimizer.lr, 0.001 * (0.98 ** temp_in))

    return history_in


# Data

raw_X = np.load('Corn_X_set_two_states.npy')
raw_y = np.load('Corn_y_set_two_states.npy')

TRAIN_MASK = np.zeros(21).astype(int)

TRAIN_MASK[:19] = True

print(TRAIN_MASK)

print('Size of feature raw set: ', raw_X.shape)
print('Size of label raw set: ', raw_y.shape)

X = np.reshape(raw_X, (COUNTY, 21, 365, 19))

print('Size of feature set: ', X.shape)
print('Size of label set: ', raw_y.shape)

X_data = X[:, TRAIN_MASK == 1, :, :]
y_data = raw_y[:, TRAIN_MASK == 1]

print('Size of feature training set: ', X_data.shape)

X_test = X[:, TRAIN_MASK == 0, :, :]
y_test = raw_y[:, TRAIN_MASK == 0]

print('Size of feature testing set: ', X_test.shape)
print('Size of label testing set: ', y_test.shape)

X_train_all = np.swapaxes(X_data, 0, 1)
print('Size of feature training set: ', X_train_all.shape)
X_train_all = np.reshape(X_train_all, (COUNTY * TRAIN_RATIO, 365, 19))
y_train_all = np.swapaxes(y_data, 0, 1)
y_train_all = np.reshape(y_train_all, -1)

print('Size of feature training set: ', X_train_all.shape)
print('Size of label training set: ', y_train_all.shape)

X_test = np.swapaxes(X_test, 0, 1)
X_test = np.reshape(X_test, (COUNTY * TEST_RATIO, 365, 19))
y_test = np.swapaxes(y_test, 0, 1)
y_test = np.reshape(y_test, -1)

print('Size of feature testing set: ', X_test.shape)
print('Size of label testing set: ', y_test.shape)

# adj

qW = Rook.from_shapefile('county_corn_two_states.shp')  # at least one edge

print(qW.cardinalities)
print(qW.max_neighbors)

adj = np.zeros((199, 199), int)

for ind in range(199):
    # adj[ind, ind] = 1
    for neighbors in qW[ind].keys():
        adj[ind, neighbors] = 1

print(adj)

# one similarity matrix for all years

static_features = X_data[:, :, 0, 7:8] + X_data[:, :, 0, 9:19]
static_features = np.reshape(static_features, (199, -1))
cos_sim = 1 - sp.distance.cdist(static_features, static_features, 'cosine')

cos_sim *= adj
for iiii in range(199):
    cos_sim[iiii, :] = cos_sim[iiii, :] / np.sum(cos_sim[iiii, :])

print(cos_sim.shape, cos_sim, len(cos_sim[cos_sim < 0.5]), np.min(cos_sim), np.sum(cos_sim, axis=1))

X_scaler = np.load('synthetic_X_scaler.npy')
Y1_scaler = np.load('synthetic_y1_scaler.npy')


def Z_norm_reverse(X, Xscaler, units_convert=1.0):
    return (X * Xscaler[1] + Xscaler[0]) * units_convert


def build_PhyModel():
    initializer = tf.keras.initializers.TruncatedNormal(stddev=0.3)  # mean=0.0, seed=None

    synthetic_x_input = tf.keras.layers.Input(shape=(365, 19))
    synthetic_gpp = tf.keras.layers.Input(shape=(365,))

    yield_train_x_input = tf.keras.layers.Input(shape=(365, 19))
    yield_train_gpp = tf.keras.layers.Input(shape=(365,))

    yield_test_x_input = tf.keras.layers.Input(shape=(365, 19))
    yield_test_gpp = tf.keras.layers.Input(shape=(365,))

    merged_input = tf.keras.layers.Concatenate(axis=0)([synthetic_x_input, yield_train_x_input, yield_test_x_input])

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
        temp_yield_train_gpp + temp_yield_train_ra[:, :, 0] + temp_yield_train_rh[:, :, 0] + temp_yield_train_nee[:, :,
                                                                                             0]))
    # yield_train_loss = K.mean(K.square(yield_train_gpp+yield_train_out_ra[:, :, 0]+yield_train_out_rh[:, :,
    # 0]+yield_train_out_nee[:, :, 0]))

    temp_yield_test_gpp = Z_norm_reverse(yield_test_gpp, X_scaler[8, :], 1.0)
    temp_yield_test_ra = Z_norm_reverse(yield_test_out_ra, Y1_scaler[0, :], 1.0)
    temp_yield_test_rh = Z_norm_reverse(yield_test_out_rh, Y1_scaler[1, :], 1.0)
    temp_yield_test_nee = Z_norm_reverse(yield_test_out_nee, Y1_scaler[2, :], 1.0)
    yield_test_loss = 0.1 * K.mean(K.square(
        temp_yield_test_gpp + temp_yield_test_ra[:, :, 0] + temp_yield_test_rh[:, :, 0] + temp_yield_test_nee[:, :, 0]))
    # yield_test_loss = K.mean(K.square(yield_test_gpp+yield_test_out_ra[:, :, 0]+yield_test_out_rh[:, :,
    # 0]+yield_test_out_nee[:, :, 0]))

    model = tf.keras.models.Model(
        inputs=[synthetic_x_input, synthetic_gpp, yield_train_x_input, yield_train_gpp, yield_test_x_input,
                yield_test_gpp],
        outputs=[synthetic_out_ra, synthetic_out_rh, synthetic_out_nee, synthetic_out_yield, yield_train_out_yield])

    model.add_loss(synthetic_loss)
    model.add_loss(yield_train_loss)
    model.add_loss(yield_test_loss)

    model.add_metric(synthetic_loss, name='synthetic_conservation_loss', aggregation='mean')
    model.add_metric(yield_train_loss, name='yield_train_conservation_loss', aggregation='mean')
    model.add_metric(yield_test_loss, name='yield_test_conservation_loss', aggregation='mean')

    return model


def custom_loss(y_true, y_pred_in):
    mask = tf.math.is_nan(y_true)
    return K.mean(tf.keras.losses.MeanSquaredError()(y_true[~mask], y_pred_in[~mask]))


model_all = build_PhyModel()

optimizer_all = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

model_all.compile(optimizer=optimizer_all,
                  loss=[REGRESSION_LOSS, REGRESSION_LOSS, REGRESSION_LOSS, REGRESSION_LOSS, custom_loss])
model_all.load_weights(CKPT_FOLDER_PATH + 'model_all')

# --------------------------
# w/ GNN

model_X_1 = global_model(cos_sim.copy())
optimizer_X_1 = keras.optimizers.Adam(learning_rate=0.001)
model_X_1.compile(optimizer=optimizer_X_1, loss=custom_loss)

model_X_1.layers[1].set_weights(model_all.layers[4].get_weights())
model_X_1.layers[2].set_weights(model_all.layers[5].get_weights())

history = gnn_train(model_X_1, X_train_all, y_train_all, adj.copy(), 30,
                    (X_test.copy(), y_test.copy()), False)

R2_BASE_TEST = gnn_rmse_testing_evaluation(model_X_1, X_test.copy(), y_test.copy(), adj.copy(), 2)

print(R2_BASE_TEST)
