import os
import re
import nltk
import numpy
import pandas
import torch
import datasets
import evaluate
import transformers


class TextClassification:
    
    def __init__(self, target_values, name="distilroberta-base", epochs=3, learning_rate=5e-5, batch_size=16, weight_decay=0.01, save=False, save_to=None, load_from=None):
        self.name = name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save = save
        self.save_to = save_to
        self.load_from = load_from
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_values = target_values
        self.pipe = None        
    
    def fit(self, df_train, df_test):
              
        id2label = dict([(i, l) for (i, l) in enumerate(self.target_values)])
        label2id = dict([(l, i) for (i, l) in enumerate(self.target_values)])
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name)
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
        def preprocess_function(examples):
            # return tokenizer(examples["text"], truncation=True, padding="max_length")
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=1024)
        df_train_encoded = datasets.Dataset.from_pandas(df_train).map(preprocess_function, batched=True)
        df_test_encoded = datasets.Dataset.from_pandas(df_test).map(preprocess_function, batched=True)

        number_of_classes = len(self.target_values)
        metric_average = "binary" if number_of_classes == 2 else "macro"
        # clf_metrics = evaluate.combine(["accuracy", "precision", "recall", "f1"])
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = numpy.argmax(logits, axis=-1)
            # predictions = (logits[:, 1] >= 0.000001).astype(bool)
            # predictions = (logits[:, 1] >= CLASSIFICATION_THRESHOLD).astype(bool)
            # return clf_metrics.compute(predictions=predictions, references=labels)
            results = {}
            results.update(accuracy_metric.compute(predictions=predictions, references=labels))
            results.update(precision_metric.compute(predictions=predictions, references=labels, average=metric_average))
            results.update(recall_metric.compute(predictions=predictions, references=labels, average=metric_average))
            results.update(f1_metric.compute(predictions=predictions, references=labels, average=metric_average))
            return results       
        
        model = transformers.AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=number_of_classes, id2label=id2label, 
                                                                                label2id=label2id, ignore_mismatched_sizes=True).to(self.device)
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
    
    def predict(self, x):
        if not self.pipe:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.name)
            model = transformers.AutoModelForSequenceClassification.from_pretrained(self.load_from)#.to(self.device) 
            self.pipe = transformers.TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)       
        predictions = self.pipe(x)
        #prediction_scores = [p[1]["score"] for p in predictions]
        postprocessed_predictions = []
        for entry in predictions:
            postprocessed_predictions.append(dict([(p["label"], p["score"]) for p in entry]))
        postprocessed_df = pandas.DataFrame(postprocessed_predictions)
        postprocessed_df["Class"] = postprocessed_df.idxmax(axis=1)
        return postprocessed_df
        

class TokenClassification:
    
    def __init__(self, target_attributes, name="roberta-base", epochs=10, learning_rate=1e-5, batch_size=16, weight_decay=0.01, save=False, save_to=None, load_from=None):
        self.name = name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.save = save
        self.save_to = save_to
        self.load_from = load_from
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_attributes = target_attributes
        self.pipe = None
        
    def fit(self, df_train, df_test):
               
        labels_list = ["O"] 
        for t in self.target_attributes:
            labels_list.extend(["B-" + t, "I-" + t])
        id2label = dict([(i, l) for (i, l) in enumerate(labels_list)])
        label2id = dict([(l, i) for (i, l) in enumerate(labels_list)])
        
        def token_annotator(tdf):
            
            def find_start_index(list1, list2):
                return next((i for i in range(len(list1) - len(list2) + 1) if list1[i:i + len(list2)] == list2), -1)
            
            nds = []
            for i, row in tdf.iterrows():
                text = str(row["text"]).lower()
                text_tokens = nltk.tokenize.word_tokenize(text)
                labels = ["O"] * len(text_tokens)
                for target_attribute in self.target_attributes:
                    annotation = str(row[target_attribute]).lower()
                    annotation_tokens = nltk.tokenize.word_tokenize(annotation)
                    start_index = find_start_index(text_tokens, annotation_tokens)
                    if start_index != -1:
                        labels[start_index] = "B-" + target_attribute
                        for i in range(len(annotation_tokens) - 1):
                            labels[start_index + i + 1] = "I-" + target_attribute 
                nds.append({"tokens": text_tokens, "labels": [labels_list.index(l) for l in labels]})
            return nds         

        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
            labels = []
            for i, label in enumerate(examples["labels"]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:  # Set the special tokens to -100.
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                        label_ids.append(label[word_idx])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        tokenizer = transformers.AutoTokenizer.from_pretrained(self.name, add_prefix_space=True, use_fast=True)
        data_collator = transformers.DataCollatorForTokenClassification(tokenizer=tokenizer)
        df_train_encoded = datasets.Dataset.from_list(token_annotator(df_train)).map(tokenize_and_align_labels, batched=True)
        df_test_encoded = datasets.Dataset.from_list(token_annotator(df_test)).map(tokenize_and_align_labels, batched=True)
             
        seqeval = evaluate.load("seqeval")
        def compute_metrics(p):
            predictions, labels = p
            predictions = numpy.argmax(predictions, axis=2)
            true_predictions = [
                [labels_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [labels_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            return {
                "accuracy": results["overall_accuracy"],
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"]
            }
        
        model = transformers.AutoModelForTokenClassification.from_pretrained(self.name, num_labels=len(labels_list), id2label=id2label, label2id=label2id).to(self.device)
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
            compute_metrics=compute_metrics,
        )
        trainer.train()
        if self.save:
            trainer.save_model(os.path.join(self.save_to, self.name))
        return trainer

    def predict(self, x):
        if not self.pipe:
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.name, add_prefix_space=True, use_fast=True)
            model = transformers.AutoModelForTokenClassification.from_pretrained(self.load_from).to(self.device)
            self.pipe = transformers.TokenClassificationPipeline(model=model, tokenizer=tokenizer, aggregation_strategy="max", device=torch.cuda.current_device())       
        predictions = self.pipe(x)
        postprocessed_predictions = []
        for p in predictions:
            postprocessed_entities = {}
            entity_scores = {}
            for entity in p:
                entity_name = entity["entity_group"]
                entity_value = entity["word"].strip()
                entity_score = entity["score"]
                if entity_name not in postprocessed_entities or entity_score > entity_scores[entity_name]:
                    postprocessed_entities[entity_name] = entity_value
                    entity_scores[entity_name] = entity_score
            postprocessed_predictions.append(postprocessed_entities)
        postprocessed_df = pandas.DataFrame(postprocessed_predictions, columns=self.target_attributes)
        return postprocessed_df