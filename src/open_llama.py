from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import pickle
from datasets import Dataset
import os


os.makedirs("data", exist_ok=True)

# Загрузка данных
with open("data/preprocessed_data.pkl", "rb") as f:
    prepared_df = pickle.load(f)

# Подготовка данных

def prepare_data(prepared_df):
    """
    Подготовка данных с учетом парафразирования
    """
    # Загрузка модели парафразера
    paraphraser = pipeline("text2text-generation", model="t5-small")

    def generate_paraphrases(text, num_return_sequences=3):
        """
        Генерация парафразов с использованием T5
        """
        results = paraphraser(
            f"paraphrase: {text}", 
            num_return_sequences=num_return_sequences, 
            max_length=50, 
            truncation=True,
            num_beams=num_return_sequences
        )
        return [result['generated_text'] for result in results]

    qa_data = []
    for _, row in prepared_df.iterrows():
        question = row['user_input']
        paraphrases = generate_paraphrases(question)
        for paraphrase in paraphrases:
            qa_data.append({
                "input": f"Вопрос: {paraphrase}",
                "output": f"Ответ: {row['bot_response']}"
            })
    return qa_data

qa_data = prepare_data(prepared_df)
dataset = Dataset.from_list(qa_data)

# Загрузка модели и токенизатора
MODEL_NAME = "openlm-research/open_llama_13b"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# Заморозка параметров, кроме последних двух слоев
for param in model.parameters():
    param.requires_grad = False
for param in model.model.layers[-2:].parameters():
    param.requires_grad = True

# Токенизация данных
max_length = 512  # Увеличить при необходимости

def tokenize_data(example):
    inputs = tokenizer(
        example["input"], 
        max_length=max_length,  
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    outputs = tokenizer(
        example["output"], 
        max_length=max_length,  
        truncation=True, 
        padding="max_length", 
        return_tensors="pt"
    )
    return {
        "input_ids": inputs["input_ids"][0],
        "labels": outputs["input_ids"][0]
    }

tokenized_dataset = dataset.map(tokenize_data)

training_args = TrainingArguments(
    output_dir="data/fine_tuned_openllama",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Уменьшить при недостатке памяти
    gradient_accumulation_steps=4,  # Использовать накопление градиентов
    num_train_epochs=5,  # Увеличил количество эпох
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    weight_decay=0.01,
    evaluation_strategy="no",  # Убираем валидацию
    # fp16=True,  # Использовать 16-битную точность для ускорения обучения
    # device_map="auto",  # Автоматически распределять модель по доступным GPU
)

# Если GPU доступен, добавьте эту строку для ускорения
model = model.to("cuda")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

trainer.save_model("data/fine_tuned_openllama")
tokenizer.save_pretrained("data/fine_tuned_openllama")

print("Fine-tuning завершён. Модель сохранена в data/fine_tuned_openllama")
