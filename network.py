from data_handling import *
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Masking, Flatten, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras import regularizers
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import CategoricalAccuracy
logging.basicConfig(datefmt='%d-%m-%y %H:%M', format='%(asctime)-15s%(levelname)s: %(message)s', level=logging.INFO)


def construct_conv1d_model(input_data_shape=282, output_data_shape=16):
    model = Sequential(name="conv1d")
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_data_shape, 1)))
    #model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_data_shape, activation='softmax'))
    return model

def construct_mlp(input_data_shape=282, output_data_shape=16):
    reg1 = l2(0.0001)
    model = Sequential(name="MLP")
    model.add(Dense(units=100, activation='relu', input_dim=input_data_shape,  kernel_regularizer=reg1, bias_regularizer=reg1, activity_regularizer=reg1))
    model.add(Dense(units=output_data_shape, kernel_initializer='glorot_uniform', activation='softmax'))
    
    #opt = keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

def construct_fully_connected_model(input_data_shape=282, output_data_shape=16):
    logging.info("[32]-[32] model")
    model = Sequential(name="simple_fully_conn")
    model.add(Dense(units=32, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

def construct_fully_connected_model_282_128_wo_compile(input_data_shape=282, output_data_shape=16):
    logging.info("[282]-[128] model")
    model = Sequential(name="simple_fully_conn")
    model.add(Dense(units=282, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))
    return model

def construct_fully_connected_model_282_128(input_data_shape=282, output_data_shape=16):
    logging.info("[282]-[128] model")
    model = Sequential(name="simple_fully_conn")
    model.add(Dense(units=282, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

def construct_fully_connected_model_1M(input_data_shape=282, output_data_shape=16):
    logging.info("[282]-[1024]-[256] model")
    model = Sequential(name="fully_conn_model")
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    #model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    #opt = keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model

def construct_fully_connected_model_1M_drop(input_data_shape=282, output_data_shape=16):
    logging.info("[282]-[1024]-[256] model")
    model = Sequential(name="fully_conn_model")
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=1024, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=output_data_shape, activation='softmax'))

    #opt = keras.optimizers.Adam(learning_rate=0.001)
    #model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model


def compile_baseline(clf):
    cat = CategoricalAccuracy()
    opt = keras.optimizers.SGD()#
    #opt = keras.optimizers.Adam(learning_rate=0.0001)
    clf.compile(loss='categorical_crossentropy', metrics=[cat], optimizer=opt)
    return clf
    
def get_baseline_model(input_data_shape=282, output_data_shape=18):
    logging.info("[282]-[1024]-[256] multilabel model")
    clf = Sequential(name="multilabel_baseline")
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(units=512, activation='relu'))
    clf.add(Dense(units=256, activation='relu'))
    clf.add(Dense(output_data_shape, activation='sigmoid'))

    return clf

def get_baseline_model_bigger(input_data_shape=282, output_data_shape=18):
    logging.info("[282]-[1024]-[512]-[256] multilabel model")
    clf = Sequential(name="multilabel_bigger_baseline")
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(units=1024, activation='relu'))
    clf.add(Dense(units=512, activation='relu'))
    clf.add(Dense(units=256, activation='relu'))
    clf.add(Dense(output_data_shape, activation='sigmoid'))

    return clf

def get_baseline_model_bigger_2(input_data_shape=282, output_data_shape=18):
    logging.info("[282]-[2048]-[1024]-[256] multilabel model")
    clf = Sequential()
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(units=2048, activation='relu'))
    clf.add(Dense(units=1024, activation='relu'))
    clf.add(Dense(units=256, activation='relu'))
    clf.add(Dense(output_data_shape, activation='sigmoid'))

    return clf

def get_simple_baseline_model(input_data_shape=282, output_data_shape=18):
    logging.info("[282]- multilabel model")
    clf = Sequential(name="multilabel_simple_model")
    clf.add(Dense(input_data_shape, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(output_data_shape, activation='sigmoid'))
    return clf

def get_simple_baseline_model_alt(input_data_shape=282, output_data_shape=18):
    logging.info("[512]-[128]- multilabel model")
    clf = Sequential(name="multilabel_simple_alt_model")
    clf.add(Dense(512, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(128, activation='relu', input_dim=input_data_shape))
    clf.add(Dense(output_data_shape, activation='sigmoid'))
    return clf

def construct_fully_connected_model_max_norm(input_data_shape=282, output_data_shape=16):
    model = Sequential()
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(Dense(units=input_data_shape//2, kernel_constraint=max_norm(3), bias_constraint=max_norm(3)))
    model.add(Dense(units=output_data_shape, kernel_constraint=max_norm(3), bias_constraint=max_norm(3), activation='softmax'))
    return model

def construct_fully_connected_model_drop(input_data_shape=282, output_data_shape=16):
    model = Sequential()
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape))
    model.add(Dropout(0.5))
    model.add(Dense(units=input_data_shape//2))
    model.add(Dropout(0.5))
    model.add(Dense(units=output_data_shape, activation='softmax'))
    return model

def construct_fully_connected_model_rfeg(input_data_shape=282, output_data_shape=16):
    model = Sequential()
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=input_data_shape//2, kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(units=output_data_shape, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    return model

def construct_fully_connected_model_small(input_data_shape=282, output_data_shape=16):
    model = Sequential()
    model.add(Dense(units=input_data_shape, activation='relu', input_dim=input_data_shape))
    model.add(Dense(units=output_data_shape, activation='softmax'))
    return model

def construct_fully_connected_vectorized_model(mask_value, shape=(5,80,), input_data_shape=282, output_data_shape=16):
    model = Sequential()
    model.add(Masking(mask_value=mask_value, input_shape=shape))
    model.add(Dense(units=input_data_shape//2))
    model.add(Flatten())
    model.add(Dense(units=output_data_shape, activation='softmax'))
    return model

def split_data_and_run_model(df, input_data_shape=282, output_data_shape=16, epochs=100, n_index=1):

    model = construct_fully_connected_model(input_data_shape, output_data_shape)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    X_train, X_val, y_train, y_val = split_data(df)
    y_train_, y_val_ = get_labels_for_index(y_train, y_val, n_index)
    history = model.fit(X_train, y_train_, epochs=epochs, validation_data=(X_val, y_val_))

    return model, history

def get_predictions(model, df):
    softmax_predictions = model.predict(df.drop(get_descriptive_col_names(), axis=1).values)
    return [np.argmax(p) for p in softmax_predictions]

def get_classification_probs_nn(x, model, value='value_l1', debug=False):
    """Collects classification probabilities for an object on 1 level

    Parameters
    ----------
    x: numpy array
        single row (without labels)
    model : RF model
    est_mapping : Dictionary
        mapping of estimator representation of the class (begins with 0) to real class labels
    value : String
        label of the level

    Returns
    -------
    list
        list of sorted dict values: label: percentage of votes
    """
    classes_votes = []
    probs_2 = model.predict(x)[0]
    probs_dict = {}; c= 0
    for p in probs_2:
        probs_dict[c] = p; c+=1
    probs_dict = {k: v for k, v in sorted(probs_dict.items(), key=lambda item: item[1], reverse=True)}

    for i in probs_dict.items():
        classes_votes.append({value: int(i[0]), 'votes_perc': i[1]})

    #classes_votes_l2 = sorted(classes_votes, key=(lambda i: i['votes_perc']), reverse=True)
    return classes_votes