import abc
import json
import os
import shutil

class AllocationDataToDifferentFolder(abc.ABC):
    """
    將資料從一大包->根據制定規則將同一類資料放在同一個資料夾
    """
    def __init__(self, source_folder_path, destination_folder_path, class_num):
        self.source_folder_path = source_folder_path
        self.destination_folder_path = destination_folder_path
        self.class_num = class_num

    def execute(self):
        """
        要把資料按照分類的資料夾存放好
        """
        return NotImplemented

class ESunHandWritingPreprocess(AllocationDataToDifferentFolder):

    def __init__(self, source_folder_path, class_num, word_dictionary_path, destination_folder_path):
        super().__init__(source_folder_path, destination_folder_path, class_num)
        self.word_dictionary_path = word_dictionary_path
        
    def execute(self):
        #讀取所有檔案列表
        filenames = os.listdir(self.source_folder_path)

        #創建資料夾(根據官方字典去建立對應的資料夾)
        vocab_dict = {}
        count = 0
        with open(self.word_dictionary_path, "r") as fp:
            for line in fp.readlines():
                line = line.replace("\n","")
                vocab_dict[line]=str(count)
                count = count+1
        for word, index in vocab_dict.items():
            complete_path = os.path.join(self.destination_folder_path,index)
            if not os.path.isdir(complete_path):
                os.mkdir(complete_path)

        #撰寫匹配規則，將檔案分配對應的資料夾
        for filename in filenames:
            if not os.path.isfile(os.path.join(os.path.join(self.destination_folder_path,vocab_dict.get(filename[-5])),filename)):
                src_complete_path = os.path.join(self.source_folder_path,filename)
                dst_complete_path = os.path.join(self.destination_folder_path,vocab_dict.get(filename[-5]))
                shutil.copy(src=src_complete_path, dst=dst_complete_path)
                print(f"add file {filename}")

if __name__ == "__main__":
    e_sun = ESunHandWritingPreprocess(source_folder_path="/Users/timshieh/Documents/HandWritingCategory/origin_data/玉山人工智慧公開挑戰賽2021夏季賽/train", 
                                      word_dictionary_path="/Users/timshieh/Documents/HandWritingCategory/origin_data/玉山人工智慧公開挑戰賽2021夏季賽/training data dic.txt",
                                      destination_folder_path="/Users/timshieh/Documents/Reconition_Github/E.SunBankOCR/data",
                                      class_num=800)
    e_sun.execute()
