## 環境建置
- pip install -r requirement.txt

## 使用方式
# 1.使用data_pipeline.py,
- 將py檔內對應的資料夾路徑都改掉,就可以自動進行將一大包資料分成多個資料夾,然後再根據分好類別的資料夾去切割train/valid/test資料集
# 2.修改training_config.yaml
- 調整內部的 TRAIN_DIR,VALID_DIR,TEST_DIR,也可以調整backbone
# 3.執行sample.py進行訓練
- 要記得修改sample.py內部的config_file變數,給入training_config.yaml進行資料匯入,模型建構,並開始訓練模型

