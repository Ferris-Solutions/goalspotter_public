import os
import numpy
import pandas
import torch
import datasets
import evaluate
import transformers


class TransformerModel:
    
    def __init__(self, name="climatebert/environmental-claims", epochs=3, learning_rate=1e-5, batch_size=16, weight_decay=0.01, save=False, save_to=None, load_from=None):
        self.name = name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save=save
        self.save_to=save_to
        self.load_from=load_from
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def fit(self, df_train, df_test):
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name)
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        def preprocess_function(examples):
            # return tokenizer(examples["text"], truncation=True, padding="max_length")
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=1024)

        df_train_encoded = datasets.Dataset.from_pandas(df_train).map(preprocess_function, batched=True)
        df_test_encoded = datasets.Dataset.from_pandas(df_test).map(preprocess_function, batched=True)

        clf_metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = numpy.argmax(logits, axis=-1)
            # predictions = (logits[:, 1] >= 0.000001).astype(bool)
            # predictions = (logits[:, 1] >= CLASSIFICATION_THRESHOLD).astype(bool)
            return clf_metrics.compute(predictions=predictions, references=labels)

        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=df_train["labels"].nunique()).to(self.device)
        training_args = transformers.TrainingArguments(
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            # ------
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            weight_decay=self.weight_decay,  
            # warmup_steps=500,
            # lr_scheduler_type="linear",
            # ------
            output_dir=self.save_to,
            optim="adamw_torch",
            save_strategy="no",
            evaluation_strategy="epoch",
            report_to="none",
            # load_best_model_at_end=True
        )

        trainer = transformers.Trainer(
            model=model,
            args=training_args,
            train_dataset=df_train_encoded,
            eval_dataset=df_test_encoded,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        if self.save:
            trainer.save_model(os.path.join(self.save_to, self.name))
        return trainer
    
    def load_pipeline(self, number_of_labels=2):
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name)
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_from, num_labels=number_of_labels)#.to(device)  
        pipe = transformers.TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)
        return pipe
        
