from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

# DATASET
__C.DATASET = ConfigurationNode()
__C.DATASET.SOURCE = "Inner"
__C.DATASET.DIR = str()
__C.DATASET.TRAIN_DIR = str()
__C.DATASET.VALID_DIR = str()
__C.DATASET.TEST_DIR =  str()
__C.DATASET.IMAGE_HEIGHT = int()
__C.DATASET.IMAGE_WIDTH = int()
__C.DATASET.CHANNELS = 3

# MODEL
__C.MODEL = ConfigurationNode()
__C.MODEL.BACKBONE    = "resnet18" #resnet34 resnet50 resnet101 resnet152
__C.MODEL.NUM_CLASSES = 801
__C.MODEL.BACKBONE_OUTPUT_SIZE = 512
__C.MODEL.EMBEDDING_SIZE = int()

# TRAINING
__C.TRAINING = ConfigurationNode()
__C.TRAINING.EPOCH             = 10
__C.TRAINING.BATCH_SIZE        = 1
__C.TRAINING.LEARNING_RATE     = 0.001
__C.TRAINING.SAVE_MODEL_DIR    = str()
__C.TRAINING.LOSS              = "balance_loss"
__C.TRAINING.SAVE_TIMES        = 1  #cannot work
__C.TRAINING.RETRAIN_MODEL_DIR = str()


# TESTING
__C.TESTING = ConfigurationNode()
__C.TESTING.BATCH_SIZE         = 100

def get_cfg_defaults():
    return __C.clone()