import torch
import transformers

MODEL_PATH = "microsoft/layoutlm-base-uncased"
TRAIN_DATA_PATH = r"data/labelled_Form_16"
ANNOTATIONS_PATH = r"data/labelled_annotations/labelled"
NUM_EPOCHS = 5
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

TOKENIZER = transformers.LayoutLMTokenizer.from_pretrained(MODEL_PATH)
LEARNING_RATE = 5e-5
with open(r"labels.txt", encoding="UTF-8") as fp:
    LABEL_LIST = [line.strip() for line in fp.readlines()]
NUM_CLASSES = len(LABEL_LIST)
MODEL_CONFIG = transformers.LayoutLMConfig(
    max_position_embeddings=768, num_labels=NUM_CLASSES
)
