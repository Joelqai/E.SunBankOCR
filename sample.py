from HandWritingReconition.config import get_cfg_defaults
from HandWritingReconition.train.classifier_trainer import ClassifierTrainer
from HandWritingReconition.train.embedding_trainer import EmbeddingTrainer
from HandWritingReconition.external import image_config 
from HandWritingReconition.evaluate import evaluation
import os

def load_config(config_file):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg

def classifier_train(cfg, alpha, beta, retrain):
    trainer = ClassifierTrainer(config=cfg, alpha=alpha, beta=beta, retrain=retrain)
    # trainer = TwoFFNNTrainer(config=cfg)
    trainer.train()

def develop_embedding_train(cfg):
    trainer = EmbeddingTrainer(config=cfg)
    trainer.train()

def check_data_positive_negative_rate(cfg):
    train_path = cfg.DATASET.TRAIN_DIR
    collect_data_size= []
    for i in range(cfg.MODEL.NUM_CLASSES):
        path = os.path.join(train_path,str(i))
        collect_data_size.append(len(os.listdir(path)))
    print(collect_data_size)
    total_num = sum(collect_data_size)
    alpha = [(total_num-class_num)/total_num for class_num in collect_data_size]
    beta = [(class_num)/total_num for class_num in collect_data_size]
    # print(alpha,beta)
    return alpha,beta


if __name__ == '__main__':
    import os
    config_file='/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/training_config.yaml'
    cfg = load_config(config_file)
    # 模型訓練
    image_config.image_height = cfg.DATASET.IMAGE_HEIGHT
    image_config.image_width = cfg.DATASET.IMAGE_WIDTH
    image_config.channels = cfg.DATASET.CHANNELS
    print(cfg)
    alpha,beta = check_data_positive_negative_rate(cfg)
    classifier_train(cfg, alpha, beta, retrain=False)
    # develop_embedding_train(cfg)

    #評估
    con_mat = evaluation(cfg)
    import json
    with open("result_224_224_train_valid_test_8_0_2_0relabel_resnet50.json","w") as fp:
        con_mat = con_mat.tolist()
        json.dump(con_mat,fp)
    


    
    # from external import config
    # config.save_model_dir = "saved_model/resnet18_saved_model/model"
    # config.dataset_dir = "dataset/"
    # from evaluate import evaluation
    # evaluation(cfg)