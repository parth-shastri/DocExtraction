import config
from transformers import LayoutLMForTokenClassification

layout_model_for_token_classification = LayoutLMForTokenClassification.from_pretrained(
    config.MODEL_PATH, config=config.MODEL_CONFIG, ignore_mismatched_sizes=True
)

if __name__ == "__main__":
    print(layout_model_for_token_classification)
    print(len(layout_model_for_token_classification.parameters()))
