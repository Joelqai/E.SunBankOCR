from __future__ import absolute_import, division, print_function
import tensorflow as tf
import math
from ..models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from ..models.classifier_layer import Classifier
from .prepare_data import generate_datasets
from ..loss import LossController


class CategoryModel(tf.keras.Model):
    
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def call(self, inputs, training=None, **kwargs):
        x = self.backbone(inputs)
        output = self.classifier(x)
        return output

#已經改成兩層輸出層
class ClassifierTrainer:

    def __init__(self, config=None, alpha=None, beta=None, retrain=False):
        self.config = config
        self.alpha = alpha
        self.beta  = beta
        self.retrain = retrain

    def get_model(self):
        classifier = Classifier(self.config.MODEL.NUM_CLASSES)
        backbone = resnet_50(self.config)
        if self.config.MODEL.BACKBONE == "resnet18":
            backbone = resnet_18(self.config)
        if self.config.MODEL.BACKBONE == "resnet34":
            backbone = resnet_34(self.config)
        if self.config.MODEL.BACKBONE == "resnet101":
            backbone = resnet_101(self.config)
        if self.config.MODEL.BACKBONE == "resnet152":
            backbone = resnet_152(self.config)
        category_model = CategoryModel(backbone=backbone, classifier=classifier)

        category_model.build(input_shape=(None, self.config.DATASET.IMAGE_HEIGHT, self.config.DATASET.IMAGE_WIDTH, self.config.DATASET.CHANNELS))
        category_model.summary()
        return category_model

    @tf.function
    def train_step(self, images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    # @tf.function
    def valid_step(self, images, labels, model, loss_object, valid_loss, valid_accuracy):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)
        print("valid labels     ", labels)
        print("valid predictions", tf.math.argmax(predictions,axis=1))

    def train(self):
        # GPU settings
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # if gpus:
        #     for gpu in gpus:
                # tf.config.experimental.set_memory_growth(gpu, True)
                # print("=====")
                # print("get_gpu")
                # print("=====")
            # get the original_dataset
        
        train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(self.config)
         # create model
        model = self.get_model()
        if self.retrain:
            model.load_weights(filepath=self.config.TRAINING.RETRAIN_MODEL_DIR)


        # define loss and optimizer
        loss_controller = LossController(loss_fn=self.config.TRAINING.LOSS, alpha=self.alpha, beta=self.beta, class_num=self.config.MODEL.NUM_CLASSES)
        loss_object     = loss_controller.get()
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(lr=self.config.TRAINING.LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
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
                # print(labels)
                step += 1
                self.train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        self.config.TRAINING.EPOCH,
                                                                                        step,
                                                                                        math.ceil(train_count / self.config.TRAINING.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            
            if (epoch+1)%math.ceil(self.config.TRAINING.EPOCH/self.config.TRAINING.SAVE_TIMES)==0:
                for valid_images, valid_labels in valid_dataset:
                    self.valid_step(valid_images, valid_labels, model, loss_object, valid_loss, valid_accuracy)
            info = self.config.TRAINING.SAVE_MODEL_DIR.split("/")[-2]
            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    self.config.TRAINING.EPOCH,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()), file=open(f"{self.config.MODEL.BACKBONE}_{self.config.TRAINING.LOSS}_{self.config.TRAINING.EPOCH}_{info}.txt", "a+"))

            # if (epoch+1)%math.ceil(self.config.TRAINING.EPOCH/self.config.TRAINING.SAVE_TIMES) == 0:
            #     temp = self.config.TRAINING.SAVE_MODEL_DIR
            #     temp_file = temp+str(epoch+1)+"epoch"
            #     print(temp_file)
            #     model.save_weights(filepath=temp_file, save_format='tf')

        model.save_weights(filepath=self.config.TRAINING.SAVE_MODEL_DIR, save_format='tf')
