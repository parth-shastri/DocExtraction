import config
import numpy as np
import torch
from PIL import ImageDraw, ImageFont
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

num_epochs = config.NUM_EPOCHS
label_map = {i: label for i, label in enumerate(config.LABEL_LIST)}
color_map = {}


def eval_fn(model, eval_dataloader, device):
    label_map = {i: label for i, label in enumerate(config.LABEL_LIST)}

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    # put model in evaluation mode
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():

            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]

            # forward pass
            outputs = model(**batch)
            # get the loss and logits
            tmp_eval_loss = outputs.loss
            logits = outputs.logits

            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # compute the predictions
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, labels.detach().cpu().numpy(), axis=0
                )

    # compute average evaluation loss
    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != -100:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }
    print(results)


def train_fn(model, dataloader, optimizer, device):
    model.to(device)
    model.train()
    losses = []
    for epoch in range(num_epochs):
        for i, batch in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Training Epoch {}".format(epoch + 1),
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            losses.append(loss)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        print("Loss after EPOCH {}: {:.2}".format(epoch + 1, sum(losses) / len(losses)))


def draw_boxes(img, boxes, labels):
    drawer = ImageDraw.Draw(img, "RGBA")
    font = ImageFont.load_default()

    if labels is not None:
        for box, label in zip(boxes, labels):
            drawer.rectangle(box, width=2, outline="red")
            drawer.text(
                (box[0] + 10, box[1] - 10),
                label_map[label],
                fill=color_map[label_map[label]],
                font=font,
            )
    else:
        for box in boxes:
            drawer.rectangle(box, width=2, outline="red")

    return img
