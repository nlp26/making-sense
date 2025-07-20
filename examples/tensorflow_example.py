import tensorflow as tf


def train_tensorflow_model(num_samples: int = 1000, input_dim: int = 10,
                           num_classes: int = 3, epochs: int = 20):
    """Train a simple neural network using TensorFlow/Keras."""
    X = tf.random.normal((num_samples, input_dim))
    y = tf.random.uniform((num_samples,), maxval=num_classes, dtype=tf.int32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(X, y, epochs=epochs, verbose=0)
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"Training accuracy: {acc:.2%}")


if __name__ == "__main__":
    train_tensorflow_model()
