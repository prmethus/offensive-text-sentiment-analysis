import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_addons as tfa
import keras_tuner as kt

# Creating a model with TemsprFlow's Sequential API for offensive text sentiment analysis
def create_and_compile_model(hp):

    embedding_dim = hp.Int("Embedding dim", min_value=64, max_value=512, step=64)
    dense_1_nodes = hp.Int("dense_1_nodes", min_value=8, max_value=128, step=8)

    inputs = tf.keras.layers.Input(shape=(max_length))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(
        inputs
    )
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(dense_1_nodes, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    learning_rate = hp.Choice("learning_rate", [0.001, 0.0001])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tfa.metrics.F1Score(num_classes=1, threshold=0.5),
        ],
    )

    return model


def hyperparameter_tuning(
    padded_training_sequences, labels_for_training, validation_data
):

    tuner = kt.Hyperband(
        create_and_compile_model,
        objective=kt.Objective("f1_score", direction="max"),
        max_epochs=10,
        factor=10,
        directory="kt_dir",
        project_name="otsa_",
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score", min_delta=0.0001, patience=5
        )
    ]

    tuner.search(
        padded_training_sequences,
        labels_for_training,
        validation_data=(padded_validation_sequences, labels_for_validation),
        epochs=50,
        callbacks=callbacks,
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)

    return model, tuner

def train():
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_f1_score", min_delta=0.0001, patience=5
        )
    ]
    num_epochs = 30
    batch_size = 64

    history = model.fit(padded_training,labels_for_training, epochs=num_epochs, batch_size=batch_size,verbose=1, callbacks=callbacks)
    
    return model