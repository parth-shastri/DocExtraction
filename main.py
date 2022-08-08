import config
from dataloader import csv_dataloader
from engine import eval_fn, train_fn
from model import layout_model_for_token_classification
from torch.optim import AdamW
from transformers import LayoutLMForTokenClassification


def run():
    device = config.DEVICE
    model = layout_model_for_token_classification
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    train_fn(model, csv_dataloader, optimizer, device)


if __name__ == "__main__":
    run()
