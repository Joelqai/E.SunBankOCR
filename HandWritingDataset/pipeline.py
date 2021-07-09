import os 
from .split_dataset import SplitDataset


def main():
        



    # data preprocessing

    # ...

    # data split
    dataset_dir = None
    saved_dataset_dir = None

    split_dataset = SplitDataset(dataset_dir=dataset_dir,
                                 saved_dataset_dir=saved_dataset_dir,
                                 train_ratio=0.8, 
                                 test_ratio=0.15,
                                 show_progress=True)
    split_dataset.start_splitting()

if __name__ == "__main__":
    main()