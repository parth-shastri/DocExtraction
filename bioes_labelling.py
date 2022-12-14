import glob
import os

import config
import pandas as pd


def label_bioes(ann_dir, ann_type="csv", convert_csv=True):
    ann_paths = glob.glob(os.path.join(ann_dir, "*.{}".format(ann_type)))

    for path in ann_paths:
        words = []
        labels = []
        if ann_type == "csv":
            ann_df = pd.read_csv(path)
        elif ann_type == "xlsx":
            ann_df = pd.read_excel(path)
        else:
            raise NotImplementedError("Not implemented {}".format(ann_type))
        ann_cols = ["left", "top", "width", "height", "text", "label"]
        ann_fields = ann_df[ann_cols]
        for i, row in ann_fields.iterrows():
            words.append(row["text"].strip())
            labels.append(row["label"])

        bioes_labels = []
        i = 0
        while i < len(labels):
            label = labels[i]
            # word = words[i]
            sent_len = 1
            word_pointer = i
            while word_pointer + 1 < len(labels) and label == labels[word_pointer + 1]:
                sent_len += 1
                word_pointer += 1
            if sent_len == 1:
                bioes_labels.append("S-" + label.upper() if label != "other" else "O")
            elif sent_len > 1:
                bioes_labels.append(
                    "B-" + labels[i].upper() if label != "other" else "O"
                )
                while i < word_pointer - 1:
                    bioes_labels.append(
                        "I-" + labels[i].upper() if label != "other" else "O"
                    )
                    i += 1
                bioes_labels.append(
                    "E-" + labels[i].upper() if label != "other" else "O"
                )
                i += 1
            i += 1

        ann_df["LABELS"] = bioes_labels
        if ann_type == "csv" or convert_csv:
            ann_df.to_csv(path)
        elif ann_type == "xlsx":
            ann_df.to_excel(path)


def get_label_list(ann_dir):
    ann_paths = glob.glob(os.path.join(ann_dir, "*.csv"))
    all_labels = []
    for path in ann_paths:
        labels = []
        ann_df = pd.read_csv(path)
        ann_cols = ["left", "top", "width", "height", "text", "label", "LABELS"]
        ann_fields = ann_df[ann_cols]
        for i, row in ann_fields.iterrows():
            labels.append(row["LABELS"])
        all_labels.extend(labels)

    unique_labels = set(all_labels)
    print(unique_labels)
    with open("label_bioes.txt", "w") as fp:
        for label in unique_labels:
            fp.write(label + "\n")


if __name__ == "__main__":
    label_bioes(ann_dir=config.ANNOTATIONS_PATH, ann_type="csv", convert_csv=True)

    get_label_list(ann_dir=config.ANNOTATIONS_PATH)
