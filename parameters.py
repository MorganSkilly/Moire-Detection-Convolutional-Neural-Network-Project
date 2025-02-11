import os

class Hyperparameters:

    def __init__(self):
        self.IMAGE_WIDTH = 512
        self.IMAGE_HEIGHT = 512
        self.SEED = 70
        self.NUM_EPOCHS = 20
        self.BATCH_SIZE = 20
        self.IMAGE_SIZE = (512, 512)

    def print_hyperparameters(self):
        print("\n=============================================")
        print("HYPERPARAMETERS\n")
        print(f"image size: {self.IMAGE_SIZE}")
        print(f"seed: {self.SEED}")
        print(f"epoch count: {self.NUM_EPOCHS}")
        print(f"batch size: {self.BATCH_SIZE}")

class Directories:
        
    def __init__(self):
        self.DATASET_DIRECTORY = "Dataset"
        self.TRAIN_DIRECTORY = "Dataset/train/train/all"
        self.TEST_DIRECTORY = "Dataset/test_origin/test_origin"

    def print_directories(self):
        print("\n=============================================")
        print("DATASET DIRECTORIES\n")
        print(f"dataset: {self.DATASET_DIRECTORY}")
        print(f"training data: {self.TRAIN_DIRECTORY}")
        print(f"testing data: {self.TEST_DIRECTORY}")
        print("\n=============================================")

    def walk_through_dir(self):
        for dirpath, dirnames, filenames in os.walk(self.DATASET_DIRECTORY):
            print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")
        