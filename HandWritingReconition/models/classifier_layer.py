import tensorflow as tf

class Classifier(tf.keras.layers.Layer):
    def __init__(self, class_num):
        super().__init__()
        self.fc     = tf.keras.layers.Dense(units=class_num, activation=tf.keras.activations.softmax)

    def call(self, inputs, training=None, **kwargs):
        outputs = self.fc(inputs)
        return outputs
