import tensorflow as tf
from ..train.prepare_data import generate_datasets
from ..train.classifier_trainer import ClassifierTrainer as Trainer
from ..external import image_config


def load_and_preprocess_image(img_path):
    print("image_height",image_config.image_height)
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=image_config.channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_config.image_height, image_config.image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img




def test_feature_map(image, config):
    if len(image.shape) == 3:
        image = tf.expand_dims(image, 0)
    trainer = Trainer(config)
    model = trainer.get_model()
    model.load_weights(filepath=config.TRAINING.RETRAIN_MODEL_DIR)
    predictions = model(image, training=False)
    print(predictions)
    print(model.layers)
    print(model.layers[0].output, model.layers[0].output.shape)
    outputs = [layer.output for layer in model.layers] 
    print("outputs",outputs)

def test_step(images, labels, model, loss_object, test_loss, test_accuracy):
    predictions = model(images, training=False)
    model.get
    t_loss = loss_object(labels, predictions)
    argmax_predictions = tf.argmax(predictions, axis=1)
    print(argmax_predictions, labels, t_loss)
    test_loss(t_loss)
    test_accuracy(labels, predictions)
    return labels, argmax_predictions

def evaluation(config):
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    print("開始進行測試~~~~~")
    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets(config)
    print("test",test_dataset, test_count)
    # load the model
    trainer = Trainer(config)
    model = trainer.get_model()
    model.load_weights(filepath=config.TRAINING.SAVE_MODEL_DIR)
    weights_list = model.get_weights()
    # print(weights_list[-2].shape,weights_list[-3].shape,weights_list[-1])
    
    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    collect_predict, collect_label = [],[]
    count = 0
    for test_images, test_labels in test_dataset:
        labels, argmax_predictions = test_step(test_images, test_labels, model, loss_object, test_loss, test_accuracy)
        # loss應該是一直update 會把數據一直疊加
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))
        collect_label.extend(labels)
        collect_predict.extend(argmax_predictions)
    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))
    
    con_mat = tf.math.confusion_matrix(labels=collect_label, predictions=collect_predict).numpy()
    print(con_mat,con_mat.shape)
    print(len(collect_predict), collect_predict[-30:],collect_predict.count(800), collect_label.count(800))
    from collections import Counter
    collect_label = [i.numpy() for i in collect_label]
    collect_predict = [i.numpy() for i in collect_predict]
    print(Counter(collect_label))
    print(Counter(collect_predict))
    return con_mat

