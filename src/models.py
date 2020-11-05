import tensorflow as tf


def get_mlp(exit_dim: int,
            neurons: int,
            dropout_rate: float,
            activation) -> tf.keras.Sequential:
    clf = tf.keras.Sequential()
    clf.add(tf.keras.layers.Dense(neurons, activation))
    clf.add(tf.keras.layers.Dropout(dropout_rate))
    clf.add(tf.keras.layers.Dense(neurons, activation))
    clf.add(tf.keras.layers.Dropout(dropout_rate))
    clf.add(tf.keras.layers.Dense(neurons, activation))
    clf.add(tf.keras.layers.Dropout(dropout_rate))
    clf.add(tf.keras.layers.Dense(exit_dim, activation="sigmoid"))

    return clf


def get_rnn(exit_dim: int,
            neurons: int, dropout_rate: float):
    clf = tf.keras.Sequential()
    clf.add(tf.keras.layers.GRU(neurons, return_sequences=True))
    clf.add(tf.keras.layers.Dropout(dropout_rate))
    clf.add(tf.keras.layers.GRU(neurons))
    clf.add(tf.keras.layers.Dropout(dropout_rate))
    clf.add(tf.keras.layers.Dense(exit_dim, activation="sigmoid"))
    return clf


def get_callbacks(fname: str):
    lr_reduce_ck = tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-7)
    save_ck = tf.keras.callbacks.ModelCheckpoint(
        fname, save_best_only=True, monitor='val_loss', mode='min'
    )
    es_ck = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=5, verbose=1,
        restore_best_weights=True
    )
    return [lr_reduce_ck, save_ck, es_ck]
