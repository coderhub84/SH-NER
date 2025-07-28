from transformers import AutoModelForTokenClassification

def load_model(model_name, label_list):
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id
    )
    
    return model, label2id, id2label