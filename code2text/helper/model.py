import tensorflow as tf

def tensors_dataset(dataset, batch=16):
    return (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(dataset["code"].values, tf.string),
            tf.cast(dataset["docstring"].values, tf.string)
        )
    )
).shuffle(len(dataset["code"])).batch(batch, drop_remainder=True).cache().prefetch(tf.data.AUTOTUNE)

class BatchLogs(tf.keras.callbacks.Callback):
  def __init__(self, key):
    self.key = key
    self.logs = []

  def on_train_batch_end(self, n, logs):
    self.logs.append(logs[self.key])