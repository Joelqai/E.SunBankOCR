from __future__ import absolute_import, division, print_function
import tensorflow as tf
import math
from ..models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from .prepare_data import generate_datasets
from .classifier_trainer import ClassifierTrainer


class CategoryEmbedding(tf.keras.Model):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def call(self, inputs, training=None, **kwargs):
        x = self.backbone(inputs)
        output = tf.math.l2_normalize(x)
        return output

class EmbeddingTrainer(ClassifierTrainer):

    def __init__(self, config=None):
        self.config = config

    def get_model(self):
        backbone = resnet_50(self.config)
        if self.config.MODEL.BACKBONE == "resnet18":
            backbone = resnet_18(self.config)
        if self.config.MODEL.BACKBONE == "resnet34":
            backbone = resnet_34(self.config)
        if self.config.MODEL.BACKBONE == "resnet101":
            backbone = resnet_101(self.config)
        if self.config.MODEL.BACKBONE == "resnet152":
            backbone = resnet_152(self.config)
        category_embedding_model = CategoryEmbedding(backbone=backbone)

        category_embedding_model.build(input_shape=(None, self.config.DATASET.IMAGE_HEIGHT, self.config.DATASET.IMAGE_WIDTH, self.config.DATASET.CHANNELS))
        category_embedding_model.summary()
        return category_embedding_model

    def get_model(self):
        model = resnet_50(self.config)
        if self.config.MODEL.BACKBONE == "resnet18":
            model = resnet_18(self.config)
        if self.config.MODEL.BACKBONE == "resnet34":
            model = resnet_34(self.config)
        if self.config.MODEL.BACKBONE == "resnet101":
            model = resnet_101(self.config)
        if self.config.MODEL.BACKBONE == "resnet152":
            model = resnet_152(self.config)
        model.build(input_shape=(None, self.config.DATASET.IMAGE_HEIGHT, self.config.DATASET.IMAGE_WIDTH, self.config.DATASET.CHANNELS))
        model.summary()
        return model

    @tf.function
    def train_step(self, images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(self, images, labels, model, loss_object, valid_loss, valid_accuracy):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)

    def train(self):
        # GPU settings
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # get the original_dataset
        
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(self.config)
         # create model
        model = self.get_model()
        # model.load_weights(filepath=self.config.TRAINING.SAVE_MODEL_DIR)
        weights_list = model.get_weights()
        print(weights_list[-1].shape,weights_list[-1])
        print(weights_list[-2].shape)
        print(weights_list[-3].shape)
        print("layer",len(weights_list[-1]),len(weights_list[-2]))
        print(model.weights[-1])
        for i, weights in enumerate(weights_list):
            model.layers[i].set_weights(weights)
        """
        # for i, weights in enumerate(weights_list[0:9]):
        #     model.layers[i].set_weights(weights)
        model.load_weights(filepath=self.config.TRAINING.SAVE_MODEL_DIR)
        # define loss and optimizer
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # optimizer = tf.keras.optimizers.Adadelta()

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

        # start training
        for epoch in range(self.config.TRAINING.EPOCH):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for images, labels in train_dataset:
                print(labels)
                step += 1
                self.train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        self.config.TRAINING.EPOCH,
                                                                                        step,
                                                                                        math.ceil(train_count / self.config.TRAINING.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            for valid_images, valid_labels in valid_dataset:
                self.valid_step(valid_images, valid_labels, model, loss_object, valid_loss, valid_accuracy)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    self.config.TRAINING.EPOCH,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()))

        model.save_weights(filepath=self.config.TRAINING.SAVE_MODEL_DIR, save_format='tf')
        """
if __name__ == '__main__':
    from HandWritingCategory.config import get_cfg_defaults
    def load_config(config_file):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(config_file)
        cfg.freeze()
        return cfg
    config_file="HandWritingCategory/config/experiments/exp_config_category.yaml"
    cfg = load_config(config_file)
    print(cfg)
    # trainer = CategoryEmbedding(backbone=)
    # CategoryEmbedding