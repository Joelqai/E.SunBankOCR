import os 
from HandWritingDataset.add_handmake_data import DataAugment
from HandWritingDataset.data_preprocess import ESunHandWritingPreprocess,ESunHandWritingPatchPreprocess
from HandWritingDataset.split_dataset import SplitDataset


def main(patch_training_data):
    
    # data preprocessing:將一整包資料按照類別分成對應到資料夾(資料夾皆用數字編號)
    e_sun = ESunHandWritingPreprocess(source_folder_path="/Users/timshieh/Documents/HandWritingCategory/origin_data/玉山人工智慧公開挑戰賽2021夏季賽/train", 
                                      word_dictionary_path="/Users/timshieh/Documents/HandWritingCategory/origin_data/玉山人工智慧公開挑戰賽2021夏季賽/training data dic.txt",
                                      destination_folder_path="/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/data",
                                      class_num=800)
    e_sun.execute()

    # data split(將資料分成train/valid/test三包)
    # dataset_dir會跟上面的destination_folder_path相同
    dataset_dir = '/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/data'
    saved_dataset_dir = '/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/train_data'

    split_dataset = SplitDataset(dataset_dir=dataset_dir,
                                 saved_dataset_dir=saved_dataset_dir,
                                 train_ratio=0.7, 
                                 test_ratio=0.15,
                                 show_progress=True)
    split_dataset.start_splitting()
    
    #是否要融入手作資料 將train跟patch資料整併
    if patch_training_data:
        #將原始資料影像處理成訓練資料
        source_folder = "/Users/timshieh/Documents/HandWritingCategory/cleaned_data_50_50"
        data_augment = DataAugment(source_folder=source_folder, destination_folder='/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/patch_data')
        data_augment.pipeline()
        #將資料切割1-800資料夾
        e_sun = ESunHandWritingPatchPreprocess(source_folder_path="/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/patch_data", 
                                        word_dictionary_path="/Users/timshieh/Documents/HandWritingCategory/origin_data/玉山人工智慧公開挑戰賽2021夏季賽/training data dic.txt",
                                        destination_folder_path="/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/train_data/patch",
                                        class_num=800)
        e_sun.execute()







if __name__ == "__main__":
    patch_training_data = True
    main(patch_training_data)