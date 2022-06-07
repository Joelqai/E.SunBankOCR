## 使用方式
# 1.先把資料分成數個class的資料夾,然後使用HandWritingDataset的pipeline.py進行資料前處理並切割train, valid, test資料集
# 2.修改HandWritingReconition/config/experiments/exp_config_classifier.yaml內部的 TRAIN_DIR,VALID_DIR,TEST_DIR,也可以調整backbone
# 3.執行sample.py進行訓練,會根據給入的yaml進行資料匯入,模型建構,並開始訓練模型
## by 謝皇廷Tim

## by Joel
## 補充報告
# 1. 補圖補成黑色，結果降低準度
# 2. 自己親筆寫字測試程式碼做特徵工程(人工挑字)
# 3. 最後還自己做了上傳功能的小工具


## 玉山人工智慧挑戰賽2021夏季賽(已結束)

![image](https://github.com/Joelqai/E.SunBankOCR/blob/master/HandWritingDataset/cmp14_Evaluation_1.png)
![image](https://github.com/Joelqai/E.SunBankOCR/blob/master/HandWritingDataset/cmp14_Evaluation_2.png))
