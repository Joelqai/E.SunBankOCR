import os 
from HandWritingDataset.data_preprocess import ESunHandWritingPreprocess
from HandWritingDataset.split_dataset import SplitDataset


def main():
        



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
                                 train_ratio=0.8, 
                                 test_ratio=0.15,
                                 show_progress=True)
    split_dataset.start_splitting()

if __name__ == "__main__":
    main()