import tensorflow as tf

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])

    history = model.fit(window.train, epochs=500,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history