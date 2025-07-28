import torch
from datasets import Dataset
from transformers import AutoTokenizer

class NERDataset(Dataset):
    def __init__(self, df, tokenizer, label2id):
        self.inputs = tokenizer(
            df['words'].tolist(),
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=264,
            return_tensors='pt'
        )

        labels = []
        for i, label_seq in enumerate(df['labels']):
            word_ids = self.inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label2id[label_seq[word_idx]])
                else:
                    label_ids.append(label2id[label_seq[word_idx]])
                previous_word_idx = word_idx
            labels.append(label_ids)

        self.inputs['labels'] = torch.tensor(labels)

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.inputs.items()}