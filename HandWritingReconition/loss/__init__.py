import tensorflow as tf
from enum import Enum
from .balance_loss import BalanceLoss

class LossController:
    class Type(Enum):
        sparse_category_loss = "sparse_category_loss"
        balance_loss = "balance_loss"
    
    def __init__(self, loss_fn, alpha=None, beta=None, class_num=None):
        try:
            loss = self.Type(loss_fn)
        except ValueError:
            raise Exception(f"your input is {loss_fn}, please input {[Type.value for Type in self.Type]}")

        if loss == self.Type.sparse_category_loss:
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
            print("SparseCategoricalCrossentropy")
        elif loss == self.Type.balance_loss:
            self.loss = BalanceLoss(alpha,beta,class_num)
            print("balance_loss")

    def get(self):
        return self.loss