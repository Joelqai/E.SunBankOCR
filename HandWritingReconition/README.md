## 使用方式
1.先把資料分成數個class的資料夾,放在







# 資料校正
觀察前三名有沒有判斷出來而不是第一個沒判斷出來我就認為他是錯的
https://zhuanlan.zhihu.com/p/76454995
https://1fly2sky.wordpress.com/2017/12/21/%E5%A4%9A%E5%88%86%E9%A1%9E%E5%95%8F%E9%A1%8C%E7%9A%84%E6%A8%A1%E5%9E%8B%E8%A9%95%E4%BC%B0%E6%8C%87%E6%A8%99/


# 訓練
資料擴增
使用backbone原始權重 不要自己tune

# 測試
測試各種圖片預測效果,看能不能挑出有問題的圖片  
製作confusion matrix 觀察分類狀態
製作embedding把大家距離拉遠



#實驗1

把relu抽掉
Layer (type)                 Output Shape              Param #
=================================================================
res_net_type_i (ResNetTypeI) multiple                  11190912
_________________________________________________________________
classifier (Classifier)      multiple                  410913
=================================================================
Total params: 11,601,825##
Trainable params: 11,592,225
Non-trainable params: 9,600
______________________________________________________________

balance loss
Layer (type)                 Output Shape              Param #
=================================================================
res_net_type_i (ResNetTypeI) multiple                  11190912
_________________________________________________________________
classifier (Classifier)      multiple                  410913
=================================================================
Total params: 11,601,825
Trainable params: 11,592,225
Non-trainable params: 9,600
_________________________________________________________________
balance_loss



#實驗3
224*224
train:valid:test->6:2:2
(train)5萬訓練 lr:0.001 30epoch->(valid+train(7萬訓練)) lr:0.0005 10epoch