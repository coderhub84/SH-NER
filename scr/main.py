from config import MODEL_NAME, TRAIN_FILE, TEST_FILE
from data_utils import load_data, split_data
from dataset import NERDataset
from model import load_model
from metrics import compute_metrics
from train import train_model
from transformers import AutoTokenizer

def main():
    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    assert (train_df['words'].str.len() == train_df['labels'].str.len()).all()
    assert (test_df['words'].str.len() == test_df['labels'].str.len()).all()

    train_df, eval_df = split_data(train_df)

    label_list = sorted({label for sublist in train_df['labels'] for label in sublist})
    model, label2id, id2label = load_model(MODEL_NAME, label_list)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_prefix_space=True)

    train_dataset = NERDataset(train_df, tokenizer, label2id)
    eval_dataset = NERDataset(eval_df, tokenizer, label2id)
    test_dataset = NERDataset(test_df, tokenizer, label2id)

    trainer, eval_results = train_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        lambda p: compute_metrics(p, id2label)
    )

    print("Evaluation Results:", eval_results)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    print("Test Results:", test_metrics)

if __name__ == "__main__":
    main()
