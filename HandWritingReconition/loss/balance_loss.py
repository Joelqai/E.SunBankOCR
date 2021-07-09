
import tensorflow as tf


class BalanceLoss:

    def __init__(self, alpha, beta, class_num):
        self.alpha = alpha
        self.beta  = beta
        self.class_num = class_num

    def __call__(self, y_true, y_pred):

        depth = self.class_num
        y_true = tf.one_hot(y_true, depth)  
        alpha = tf.constant(self.alpha, dtype=tf.float32)
        beta  = tf.constant(self.beta, dtype=tf.float32) 
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        # y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        print(tf.multiply(y_true,tf.math.log(tf.clip_by_value(y_pred,1e-10,1.0))).shape)
        pos = tf.multiply(alpha,tf.multiply(y_true,tf.math.log(tf.clip_by_value(y_pred,1e-10,1))))
        neg = tf.multiply(beta,tf.multiply(tf.subtract(1.,y_true),tf.math.log(tf.clip_by_value(tf.subtract(1.,y_pred),1e-10,1.0))))
        print(pos)
        print(neg)
        return -tf.reduce_sum(pos+neg)