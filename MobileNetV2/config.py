
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IN_CHANNELS = 3  #  for RGB images
IMG_HEIGHT = 224
IMG_WIDTH = 224

NUM_EPOCHS = 60
BATCH_SIZE = 32
NUM_CLASSES = 10 # instead of 1000 in the paper because they trained on imageNet dataset

LEARNING_RATE = 1e-2

DATASET_HOME = "natural_images_sets/"
LOGS_PATH = "logs"
CHECKPOINTS_DIR = "/content/drive/MyDrive/dl_projects/MobileNetV2"

SAVE_MODEL = True
LOAD_MODEL = False
