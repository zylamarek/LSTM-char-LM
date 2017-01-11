from __future__ import print_function, division
import numpy as np
import theano
import theano.tensor as T
import os
import time
import lasagne
import scipy.io as sio
import pickle
import logging
import argparse

# PARSE SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument("-BATCH_SIZE", type=int, default=100)
parser.add_argument("-DATA", type=str, default='warpeace_input.txt')
parser.add_argument("-MODEL_SEQ_LEN", type=int, default=100)
parser.add_argument("-TOL", type=float, default=1e-6)
parser.add_argument('-REC_NUM_UNITS', nargs='+', type=int,
                    default=[37, 37, 37])
parser.add_argument('-TRAIN_FRACTION', type=float, default=0.8)
parser.add_argument('-VAL_FRACTION', type=float, default=0.1)
parser.add_argument("-DROPOUT_FRACTION", type=float, default=0.08)
parser.add_argument("-LEARNING_RATE", type=float, default=2e-3)
parser.add_argument("-DECAY", type=float, default=0.95)
parser.add_argument("-NO_DECAY_EPOCHS", type=int, default=10)
parser.add_argument("-MAX_GRAD", type=float, default=5)
parser.add_argument("-NUM_EPOCHS", type=int, default=50)
parser.add_argument("-SEED", type=int, default=1234)
parser.add_argument("-INIT_RANGE", type=float, default=0.08)

args = parser.parse_args()

BATCH_SIZE = args.BATCH_SIZE
DATA = args.DATA
MODEL_SEQ_LEN = args.MODEL_SEQ_LEN
TOL = args.TOL
REC_NUM_UNITS = args.REC_NUM_UNITS
if isinstance(REC_NUM_UNITS, (int, long)):
    REC_NUM_UNITS = [REC_NUM_UNITS]
assert len(REC_NUM_UNITS) == 3
SPLIT_SIZES = [args.TRAIN_FRACTION, args.VAL_FRACTION]
DROPOUT_FRACTION = args.DROPOUT_FRACTION
LEARNING_RATE = args.LEARNING_RATE
DECAY = args.DECAY
NO_DECAY_EPOCHS = args.NO_DECAY_EPOCHS
MAX_GRAD = args.MAX_GRAD
NUM_EPOCHS = args.NUM_EPOCHS
np.random.seed(args.SEED)
INI = lasagne.init.Uniform(args.INIT_RANGE)

# CREATE OUTPUT FOLDER
folder_name = '%d_LSTM' % len(REC_NUM_UNITS)
for rnu in REC_NUM_UNITS:
    folder_name += '_%d' % rnu
folder_name = 'output/' + folder_name
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# INITIALIZE LOGGER
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(folder_name + "/train_output.log", mode='w')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

# LOG PARAMETERS
logger.info('Settings')
logger.info('BATCH_SIZE      : ' + str(BATCH_SIZE))
logger.info('DATA            : ' + str(DATA))
logger.info('MODEL_SEQ_LEN   : ' + str(MODEL_SEQ_LEN))
logger.info('TOL             : ' + str(TOL))
logger.info('REC_NUM_UNITS   : ' + str(REC_NUM_UNITS))
logger.info('SPLIT_SIZES     : ' + str(SPLIT_SIZES))
logger.info('DROPOUT_FRACTION: ' + str(DROPOUT_FRACTION))
logger.info('LEARNING_RATE   : ' + str(LEARNING_RATE))
logger.info('DECAY           : ' + str(DECAY))
logger.info('NO_DECAY_EPOCHS : ' + str(NO_DECAY_EPOCHS))
logger.info('MAX_GRAD        : ' + str(MAX_GRAD))
logger.info('NUM_EPOCHS      : ' + str(NUM_EPOCHS))
logger.info('RNG SEED        : ' + str(args.SEED))
logger.info('INIT RANGE      : ' + str(args.INIT_RANGE))
logger.info('folder_name     : ' + str(folder_name))

# GET FLOATX TYPE
floatX_dtype = np.dtype(theano.config.floatX).type

# DEFINE DATA LOADING AND PREPROCESSING FUNCTIONS
def load_data(file_name):
    vocab_size = 0
    vocab_map = {}
    with open(file_name, 'rb') as f:
        raw_data = f.read()
        x = np.zeros(raw_data.__len__())
        for iSym, symbol in enumerate(raw_data):
            if symbol not in vocab_map:
                vocab_map[symbol] = vocab_size
                vocab_size += 1
            x[iSym] = vocab_map[symbol]
    print("Loaded %i symbols from %s" % (iSym, file_name))
    return x.astype('int32'), vocab_map, vocab_size


def reorder(x_in, batch_size, model_seq_len, vocab_size):
    if x_in.shape[0] % (batch_size*model_seq_len) == 0:
        x_in = x_in[:-1]

    x_resize = (x_in.shape[0] // (batch_size*model_seq_len)) * \
               model_seq_len*batch_size
    n_samples = x_resize // model_seq_len
    n_batches = n_samples // batch_size

    targets = x_in[1:x_resize+1]
    x_out_ = x_in[:x_resize]

    x_out = np.zeros((x_out_.shape[0], vocab_size)).astype('int32')
    for iSym, symbol in enumerate(x_out_):
        x_out[iSym, symbol] = 1

    x_out = x_out.reshape((n_samples, model_seq_len, vocab_size))
    targets = targets.reshape((n_samples, model_seq_len))

    out = np.zeros(n_samples, dtype=int)
    for i in range(n_batches):
        val = range(i, n_batches*batch_size+i, n_batches)
        out[i*batch_size:(i+1)*batch_size] = val

    x_out = x_out[out]
    targets = targets[out]

    return x_out.astype(floatX_dtype), targets.astype('int32')


def get_data(model_seq_len, batch_size, split_sizes, data_file):
    x, vocab_map, vocab_size = load_data(os.path.join('data', data_file))

    n_train = np.floor(split_sizes[0] * x.shape[0])
    n_val = np.floor(split_sizes[1] * x.shape[0])
    x_train = x[:n_train]
    x_val = x[n_train: n_train+n_val]
    x_test = x[n_train+n_val:]

    x_train, y_train = \
        reorder(x_train, batch_size, model_seq_len, vocab_size)
    x_val, y_val = \
        reorder(x_val, batch_size, model_seq_len, vocab_size)
    x_test, y_test = \
        reorder(x_test, batch_size, model_seq_len, vocab_size)

    return x_train, y_train, x_val, y_val, x_test, y_test, vocab_map, vocab_size

# LOAD AND PREPROCESS DATA
t0 = time.time()
x_train, y_train, x_val, y_val, x_test, y_test, vocab_map, vocab_size = \
    get_data(MODEL_SEQ_LEN, BATCH_SIZE, SPLIT_SIZES, DATA)
logger.info('Loading and reordering data took  %.2fs' % (time.time()-t0))

# LOG DATA PROPERTIES
logger.info("-" * 80)
logger.info("Vocabulary size: " + str(vocab_size))
logger.info("Data shapes")
logger.info("Train data     : " + str(x_train.shape))
logger.info("Validation data: " + str(x_val.shape))
logger.info("Test data      : " + str(x_test.shape))
logger.info("-" * 80)

# DEFINE THEANO SYMBOLIC VARIABLES
sym_x = T.tensor3(dtype=theano.config.floatX)
sym_y = T.imatrix()
cell1_init_sym = T.matrix(dtype=theano.config.floatX)
hid1_init_sym = T.matrix(dtype=theano.config.floatX)
cell2_init_sym = T.matrix(dtype=theano.config.floatX)
hid2_init_sym = T.matrix(dtype=theano.config.floatX)
cell3_init_sym = T.matrix(dtype=theano.config.floatX)
hid3_init_sym = T.matrix(dtype=theano.config.floatX)

# BUILD THE MODEL

# INPUT
l_inp = lasagne.layers.InputLayer(
    shape=(BATCH_SIZE, MODEL_SEQ_LEN, vocab_size),
    input_var=sym_x)

# DROPOUT
l_in_reshape = lasagne.layers.ReshapeLayer(
    l_inp,
    (BATCH_SIZE*MODEL_SEQ_LEN, vocab_size))

l_drp0 = lasagne.layers.DropoutLayer(
    l_in_reshape,
    p=DROPOUT_FRACTION)

l_drp0_reshape = lasagne.layers.ReshapeLayer(
    l_drp0,
    (BATCH_SIZE, MODEL_SEQ_LEN, vocab_size))

# FIRST LSTM
l_rec1 = lasagne.layers.LSTMLayer(
    l_drp0_reshape,
    num_units=REC_NUM_UNITS[0],
    peepholes=False,
    ingate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    forgetgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    outgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    cell=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None,
                             nonlinearity=lasagne.nonlinearities.tanh),
    learn_init=False,
    cell_init=cell1_init_sym,
    hid_init=hid1_init_sym,
    precompute_input=True,
    grad_clipping=MAX_GRAD)

# DROPOUT
l_drp1 = lasagne.layers.DropoutLayer(
    l_rec1,
    p=DROPOUT_FRACTION)

# SECOND LSTM
l_rec2 = lasagne.layers.LSTMLayer(
    l_drp1,
    num_units=REC_NUM_UNITS[1],
    peepholes=False,
    ingate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    forgetgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    outgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    cell=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None,
                             nonlinearity=lasagne.nonlinearities.tanh),
    learn_init=False,
    cell_init=cell2_init_sym,
    hid_init=hid2_init_sym,
    precompute_input=True,
    grad_clipping=MAX_GRAD)

# DROPOUT
l_drp2 = lasagne.layers.DropoutLayer(
    l_rec2,
    p=DROPOUT_FRACTION)

# THIRD LSTM
l_rec3 = lasagne.layers.LSTMLayer(
    l_drp2,
    num_units=REC_NUM_UNITS[2],
    peepholes=False,
    ingate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    forgetgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    outgate=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=INI),
    cell=lasagne.layers.Gate(W_in=INI, W_hid=INI, W_cell=None,
                             nonlinearity=lasagne.nonlinearities.tanh),
    learn_init=False,
    cell_init=cell3_init_sym,
    hid_init=hid3_init_sym,
    precompute_input=True,
    grad_clipping=MAX_GRAD)

# DROPOUT
l_drp3 = lasagne.layers.DropoutLayer(
    l_rec3,
    p=DROPOUT_FRACTION)

l_shp = lasagne.layers.ReshapeLayer(
    l_drp3,
    (BATCH_SIZE*MODEL_SEQ_LEN, REC_NUM_UNITS[2]))

# SOFTMAX OUTPUT
l_out = lasagne.layers.DenseLayer(
    l_shp,
    num_units=vocab_size,
    nonlinearity=lasagne.nonlinearities.softmax)

# DEFINE LOSS FUNCTION
def cross_ent(net_output, target):
    predictions = \
        T.reshape(net_output, (BATCH_SIZE*MODEL_SEQ_LEN, vocab_size))
    predictions += TOL
    target = T.flatten(target)
    cost_ce = T.nnet.categorical_crossentropy(predictions, target)
    return T.sum(cost_ce)

# DEFINE THEANO VARIABLES FOR TRAINING AND EVALUATION
train_out = lasagne.layers.get_output(l_out, deterministic=False)
hidden_states_train = [l_rec1.cell_out, l_rec1.hid_out,
                       l_rec2.cell_out, l_rec2.hid_out,
                       l_rec3.cell_out, l_rec3.hid_out]
cost_train = cross_ent(train_out, sym_y)

eval_out = lasagne.layers.get_output(l_out, deterministic=True)
hidden_states_eval = [l_rec1.cell_out, l_rec1.hid_out,
                      l_rec2.cell_out, l_rec2.hid_out,
                      l_rec3.cell_out, l_rec3.hid_out]
cost_eval = cross_ent(eval_out, sym_y)

# GET ALL PARAMETERS OF THE NETWORK
all_params = lasagne.layers.get_all_params(l_out, trainable=True)

# LOG THE PARAMETERS
logger.info("-" * 80)
total_params = sum([p.get_value().size for p in all_params])
logger.info("#NETWORK params: " + str(total_params))
logger.info("Parameters:")
logger.info([{a.name, a.get_value().shape} for a in all_params])
logger.info("-" * 80)

# DEFINE THEANO VARIABLE FOR GRADIENT
all_grads = T.grad(cost_train, all_params)

# DEFINE UPDATE RULES
sh_lr = theano.shared(lasagne.utils.floatX(LEARNING_RATE))
updates = lasagne.updates.rmsprop(all_grads, all_params,
                                  learning_rate=sh_lr)

# DEFINE EVALUATION AND TRAIN FUNCTIONS
fun_inp = [sym_x, sym_y,
           cell1_init_sym, hid1_init_sym,
           cell2_init_sym, hid2_init_sym,
           cell3_init_sym, hid3_init_sym]

logger.info("compiling f_eval...")
f_eval = theano.function(fun_inp,
                         [cost_eval,
                          hidden_states_eval[0][:, -1],
                          hidden_states_eval[1][:, -1],
                          hidden_states_eval[2][:, -1],
                          hidden_states_eval[3][:, -1],
                          hidden_states_eval[4][:, -1],
                          hidden_states_eval[5][:, -1]],
                         allow_input_downcast=True)

logger.info("compiling f_train...")
f_train = theano.function(fun_inp,
                          [cost_train,
                           hidden_states_train[0][:, -1],
                           hidden_states_train[1][:, -1],
                           hidden_states_train[2][:, -1],
                           hidden_states_train[3][:, -1],
                           hidden_states_train[4][:, -1],
                           hidden_states_train[5][:, -1]],
                          updates=updates,
                          allow_input_downcast=True)

# DEFINE HELPER FUNCTION FOR DATASET EVALUATION
def calc_cross_entropy(x, y):
    n_batches = x.shape[0] // BATCH_SIZE
    l_cost = []
    cell1, hid1 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[0]),
                            dtype=floatX_dtype) for _ in range(2)]
    cell2, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[1]),
                            dtype=floatX_dtype) for _ in range(2)]
    cell3, hid3 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[2]),
                            dtype=floatX_dtype) for _ in range(2)]
    for i in range(n_batches):
        x_batch = x[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        cost, cell1, hid1, cell2, hid2, cell3, hid3 = f_eval(
            x_batch, y_batch, cell1, hid1, cell2, hid2, cell3, hid3)
        l_cost.append(cost)

    ce = np.mean(l_cost) / BATCH_SIZE / MODEL_SEQ_LEN
    return ce


# MAIN TRAINING LOOP
logger.info("Begin training...")
ce_train = []
ce_valid = []
ce_test = []
n_batches_train = x_train.shape[0] // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    l_cost, batch_time = [], time.time()

    # INITIALIZE HIDDEN STATE WITH ZEROS
    cell1, hid1 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[0]),
                            dtype=floatX_dtype) for _ in range(2)]
    cell2, hid2 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[1]),
                            dtype=floatX_dtype) for _ in range(2)]
    cell3, hid3 = [np.zeros((BATCH_SIZE, REC_NUM_UNITS[2]),
                            dtype=floatX_dtype) for _ in range(2)]

    # TRAIN ALL THE BATCHES
    for i in range(n_batches_train):
        # FETCH DATA
        x_batch = x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        y_batch = y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]

        # TRAIN
        cost, cell1, hid1, cell2, hid2, cell3, hid3 = f_train(
            x_batch, y_batch, cell1, hid1, cell2, hid2, cell3, hid3)
        l_cost.append(cost)

    # APPLY LEARNING RATE DECAY
    if epoch > (NO_DECAY_EPOCHS - 1):
        current_lr = sh_lr.get_value()
        sh_lr.set_value(lasagne.utils.floatX(current_lr * float(DECAY)))

    # EVALUATE AND LOG
    elapsed = time.time() - batch_time
    chars_per_second = \
        float(BATCH_SIZE*MODEL_SEQ_LEN*n_batches_train) / elapsed
    crossent_valid = calc_cross_entropy(x_val, y_val)
    crossent_train = np.mean(l_cost) / BATCH_SIZE / MODEL_SEQ_LEN
    crossent_test = calc_cross_entropy(x_test, y_test)
    logger.info("*------------ Epoch: " + str(epoch))
    logger.info("Cross entropy train: " + str(crossent_train))
    logger.info("Cross entropy valid: " + str(crossent_valid))
    logger.info("Cross entropy test : " + str(crossent_test))
    logger.info("Chars per second   : " + str(chars_per_second))
    logger.info("Time elapsed       : " + str(int(elapsed)) + ' s')
    logger.info("ETA                : " +
                str(int(elapsed*(NUM_EPOCHS-epoch-1)/60)) + ' min')

    ce_train.append(crossent_train)
    ce_valid.append(crossent_valid)
    ce_test.append(crossent_test)

    # STORE CROSS ENTROPY
    sio.savemat(folder_name + '/ce.mat', {'ce_train': ce_train,
                                          'ce_valid': ce_valid,
                                          'ce_test': ce_test})

    # STORE PARAMETERS
    pickle.dump(all_params,
                open(folder_name + '/params_' + str(epoch) + '.p', "wb"))

print("done.")
