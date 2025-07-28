from transformers import TrainingArguments, Trainer

def train_model(model, tokenizer, train_dataset, eval_dataset, compute_metrics_fn, output_dir="./ner-roberta1"):
    args = TrainingArguments(
        output_dir=output_dir,
        do_eval=True,
        do_train=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        num_train_epochs=6,
        learning_rate=2e-5,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        eval_steps=500,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer
    )

    trainer.train()
    eval_results = trainer.evaluate()
    return trainer, eval_results