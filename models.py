import tensorflow as tf


def conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    dropout: float,
    negative_slope: float,
    name_prefix: str,
) -> tf.Tensor:
    x = tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        kernel_initializer="he_normal",
        name=f"{name_prefix}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = tf.keras.layers.LeakyReLU(
        negative_slope=negative_slope,
        name=f"{name_prefix}_lrelu",
    )(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name_prefix}_drop")(x)
    return x


def build_cnn_gru_model(config) -> tf.keras.Model:
    inputs = tf.keras.Input(
        shape=(config.window_size, config.n_features),
        name=f"{config.task}_input",
    )

    x = inputs

    for i, filters in enumerate(config.conv_filters, start=1):
        x = conv_block(
            x=x,
            filters=filters,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            negative_slope=config.negative_slope,
            name_prefix=f"block{i}",
        )

    x = tf.keras.layers.GRU(
        units=config.gru_units,
        return_sequences=False,
        name="gru",
    )(x)

    x = tf.keras.layers.Dense(
        config.dense_units,
        kernel_initializer="he_normal",
        name="dense_hidden",
    )(x)
    x = tf.keras.layers.LeakyReLU(
        negative_slope=config.negative_slope,
        name="dense_hidden_lrelu",
    )(x)

    outputs = tf.keras.layers.Dense(1, activation="linear", name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{config.cohort}_{config.task}_cnn_gru")

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )

    return model