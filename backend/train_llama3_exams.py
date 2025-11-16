from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

dataset = load_dataset("json", data_files="mental_health_exams.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    text = f"User: {example['instruction']}\nAssistant: {example['response']}"
    return tokenizer(text, truncation=True, max_length=512)

dataset = dataset["train"].map(tokenize)

print("Tokenization complete.")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="float32",
    device_map={"": "cpu"}   # everything on CPU
)

print("Model loaded.")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

print("LoRA model prepared.")

training_args = TrainingArguments(
    output_dir="./exams_lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    warmup_steps=10,
    max_steps=50,            # SMALL so it runs
    learning_rate=1e-4,
    fp16=False,
    logging_steps=5,
    save_steps=50
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer)
)

trainer.train()

model.save_pretrained("./exams_lora")
print("Training finished.")
