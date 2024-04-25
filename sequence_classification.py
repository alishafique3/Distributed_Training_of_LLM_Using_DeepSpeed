import os
import deepspeed
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# Set Hugging Face API token 
os.environ["HF_TOKEN"] = "hf_**************************" 
os.environ["TOKENIZERS_PARALLELISM"] = "true" 

os.environ['WANDB_API_KEY'] = 'af***********************'
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="distributed_training"
# save your trained model checkpoint to wandb
os.environ["WANDB_LOG_MODEL"]="true"
# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


def main():
    """## Load IMDb dataset

    Start by loading the IMDb dataset from the 🤗 Datasets library:
    """

    imdb = load_dataset("imdb")

    """Then take a look at an example:"""

    print(imdb["test"][0])

    """There are two fields in this dataset:

    - `text`: the movie review text.
    - `label`: a value that is either `0` for a negative review or `1` for a positive review.

    ## Preprocess

    The next step is to load a DistilBERT tokenizer to preprocess the `text` field:
    """



    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    """Create a preprocessing function to tokenize `text` and truncate sequences to be no longer than DistilBERT's maximum input length:"""

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    """To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map
    (https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up `map` by setting `batched=True` to
    process multiple elements of the dataset at once:"""

    tokenized_imdb = imdb.map(preprocess_function, batched=True)

    """Now create a batch of examples using [DataCollatorWithPadding
    (https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithPadding). It's more efficient to *dynamically pad* the
    sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length."""



    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    """## Evaluate

    Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate
    (https://huggingface.co/docs/evaluate/index) library. For this task, load the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗
    Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):
    """

    accuracy = evaluate.load("accuracy")

    """Then create a function that passes your predictions and labels to [compute
    (https://huggingface.co/docs/evaluate/main/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the accuracy:"""
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    
    """Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.

    ## Train

    Before you start training your model, create a map of the expected ids to their labels with `id2label` and `label2id`:
    """

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    """<Tip>

    If you aren't familiar with finetuning a model with the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), take a
    look at the basic tutorial [here](https://huggingface.co/docs/transformers/main/en/tasks/../training#train-with-pytorch-trainer)!

    </Tip>

    You're ready to start training your model now! Load DistilBERT with [AutoModelForSequenceClassification
    (https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification) along with the number of expected labels, and the
    label mappings:
    """
    
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    """At this point, only three steps remain:

    1. Define your training hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments).
    The only required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be
    signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer
    (https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) will evaluate the accuracy and save the training checkpoint.2. Pass the
    training arguments to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer,
    data collator, and `compute_metrics` function.
    3. Call [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.
    """

    training_args = TrainingArguments(
        output_dir="my_awesome_model",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        #push_to_hub=True,
        deepspeed="/tmp/ds_config_zero2.json",
        report_to='wandb'
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

if __name__ == "__main__":
    main()

"""<Tip>

[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) applies dynamic padding by default when you pass `tokenizer` to it. In this case, you don't need to specify a data collator explicitly.

</Tip>

Once training is completed, share your model to the Hub with the [push_to_hub()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.push_to_hub) method so everyone can use your model:
"""

# trainer.push_to_hub()

"""<Tip>

For a more in-depth example of how to finetune a model for text classification, take a look at the corresponding
[PyTorch notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification.ipynb)
or [TensorFlow notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/text_classification-tf.ipynb).

</Tip>

## Inference

Great, now that you've finetuned a model, you can use it for inference!

Grab some text you'd like to run inference on:
"""

# text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# """The simplest way to try out your finetuned model for inference is to use it in a [pipeline()](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.pipeline). Instantiate a `pipeline` for sentiment analysis with your model, and pass your text to it:"""

# from transformers import pipeline

# classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
# classifier(text)

# """You can also manually replicate the results of the `pipeline` if you'd like:

# Tokenize the text and return PyTorch tensors:
# """

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("stevhliu/my_awesome_model")
# inputs = tokenizer(text, return_tensors="pt")

# """Pass your inputs to the model and return the `logits`:"""

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("stevhliu/my_awesome_model")
# with torch.no_grad():
#     logits = model(**inputs).logits

# """Get the class with the highest probability, and use the model's `id2label` mapping to convert it to a text label:"""

# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]
