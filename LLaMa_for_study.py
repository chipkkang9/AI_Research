from datasets import load_dataset
from transformers import TrainingArguments, LlamaConfig, LlamaForCausalLM, AutoTokenizer, Trainer

# 1. Preparing Datasets
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

# 2. Preparing Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# 3. Model Architecture
config = LlamaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_size = 256,
    intermediate_size=512,
    num_hidden_layers=4,
    num_attention_heads=4,
)

model = LlamaForCausalLM(config)

# 4. Tokenizing, Dataset Processing
def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

block_size = 128 # Model Memorization Scope

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

# 5. Training Processs Setting and Execution
training_args = TrainingArguments(
    output_dir="./llama-pretrain-debug",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    logging_strategy="steps",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets,
)

# Code run
trainer.train()