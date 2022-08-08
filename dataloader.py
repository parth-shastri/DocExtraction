import glob
import json
import os

import config
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

label_list = config.LABEL_LIST


class DocParsingDataset(Dataset):
    def __init__(
        self,
        image_path,
        anno_path,
        tokenizer,
        max_length,
        pad_label_id,
        ann_type="json",
    ):
        super(DocParsingDataset, self).__init__()
        self.image_path = image_path
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ann_type = ann_type
        self.pad_label_id = pad_label_id
        if ann_type not in ["csv", "json", "xlsx"]:
            raise NotImplementedError(
                "Not implemented pass in {}".format(["csv", "json", "xlsx"])
            )
        self.anno_filenames = sorted(
            glob.glob(os.path.join(self.anno_path, "*.{}".format(ann_type)))
        )

        self.image_names = sorted(
            glob.glob(
                os.path.join(self.image_path, "*.jpg"),
            )
        )[: len(self.anno_filenames)]

        if len(self.image_names) == 0:
            self.image_names = sorted(
                glob.glob(
                    os.path.join(self.image_path, "*.png"),
                )
            )[: len(self.anno_filenames)]

    def __len__(self):
        return len(self.anno_filenames)

    @staticmethod
    def normalize(box, img_width, img_height):
        return [
            int(1000 * box[0] / img_width),
            int(1000 * box[1] / img_height),
            int(1000 * box[2] / img_width),
            int(1000 * box[3] / img_height),
        ]

    def create_data_examples(
        self,
        words,
        labels,
        boxes,
        cls_token_box=[0, 0, 0, 0],
        sep_token_box=[1000, 1000, 1000, 1000],
        pad_token_box=[0, 0, 0, 0],
    ):
        """take in words boxes and labels return tokenized output with params
        input_ids, attention_mask, labels, and token_type_ids"""
        label_map = {label: i for i, label in enumerate(label_list)}
        pad_token = self.tokenizer.pad_token
        sep_token = self.tokenizer.sep_token
        cls_token = self.tokenizer.cls_token
        tokens = []
        token_boxes = []
        label_ids = []

        for word, label, box in zip(words, labels, boxes):
            word_tokens = self.tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            token_boxes.extend([box] * len(word_tokens))
            label_ids.extend(
                [label_map[label]] + [self.pad_label_id] * (len(word_tokens) - 1)
            )

        special_tokens_count = 2
        if len(tokens) > self.max_length - special_tokens_count:
            tokens = tokens[: (self.max_length - special_tokens_count)]
            token_boxes = token_boxes[: (self.max_length - special_tokens_count)]
            label_ids = label_ids[: (self.max_length - special_tokens_count)]

        tokens += [sep_token]
        token_boxes += [sep_token_box]
        label_ids += [self.pad_label_id]

        tokens = [cls_token] + tokens
        token_boxes = [cls_token_box] + token_boxes
        label_ids = [self.pad_label_id] + label_ids

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_len
        token_boxes += [pad_token_box] * padding_len
        attention_mask += [0] * padding_len
        label_ids += [self.pad_label_id] * padding_len

        assert len(input_ids) == self.max_length
        assert len(token_boxes) == self.max_length
        assert len(attention_mask) == self.max_length
        assert len(label_ids) == self.max_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_type_ids": torch.zeros(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "bbox": torch.tensor(token_boxes),
        }

    def __getitem__(self, item):
        ann_file = self.anno_filenames[item]
        img_file = self.image_names[item]
        # print(ann_file, img_file)
        # if item < 10:
        #     print(img_file, ann_file)
        img = Image.open(img_file).convert("RGB")
        img_height, img_width, _ = np.asarray(img).shape
        words, boxes, labels = [], [], []
        if self.ann_type == "json":
            with open(ann_file, encoding="UTF-8") as f:
                ann_data = json.load(f)

            for ann in ann_data["form"]:
                if ann["text"] != "":
                    wrd = [
                        word["text"].strip()
                        for word in ann["words"]
                        if word["text"] != ""
                    ]
                    words.extend(wrd)
                    labels.extend([ann["label"]] * len(wrd))
                    boxes.extend(
                        [
                            self.normalize(word["box"], img_width, img_height)
                            for word in ann["words"]
                        ]
                    )

        elif self.ann_type == "csv":
            ann_df = pd.read_csv(ann_file)
            ann_cols = ["left", "top", "width", "height", "text", "label"]
            ann_fields = ann_df[ann_cols]
            for i, row in ann_fields.iterrows():
                words.append(row["text"].strip())
                labels.append(row["label"])
                box = [
                    row["left"],
                    row["top"],
                    row["width"] + row["left"],
                    row["height"] + row["top"],
                ]
                boxes.append(self.normalize(box, img_width, img_height))

        elif self.ann_type == "xlsx":
            ann_df = pd.DataFrame(pd.read_excel(ann_file))

            ann_cols = ["left", "top", "width", "height", "text", "label"]
            ann_fields = ann_df[ann_cols]
            for i, row in ann_fields.iterrows():
                words.append(
                    row["text"].strip()
                    if isinstance(row["text"], str)
                    else str(row["text"]).strip()
                )
                labels.append(
                    row["label"].strip() if row["label"].strip() != "nan" else "other"
                )
                box = [
                    row["left"],
                    row["top"],
                    row["width"] + row["left"],
                    row["height"] + row["top"],
                ]
                boxes.append(self.normalize(box, img_width, img_height))

        else:
            raise NotImplementedError("Use ann_type as " "json" " or " "csv" "!")

        annotations = self.create_data_examples(words, labels, boxes)

        return annotations


csv_dataset = DocParsingDataset(
    config.TRAIN_DATA_PATH, config.ANNOTATIONS_PATH, config.TOKENIZER, 768, -100, "csv"
)

csv_dataloader = DataLoader(csv_dataset, batch_size=2, shuffle=True)

if __name__ == "__main__":
    tokenizer = config.TOKENIZER
    for tokens in tokenizer.convert_ids_to_tokens(csv_dataset[0]["input_ids"]):
        print(tokens, end=" ")
