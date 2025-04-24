# RAG-система на FastAPI + Milvus + LLaMA

![Architecture](assets/rag_architecture.png)

---

## Описание

Это система вопрос-ответ (Q&A) по вопросам агрегатора маркетплейсов для селлеров, MVP модель - для использования затереть из predict.py строку с вопросом. Модель позволяет пользователям задавать вопросы, на которые система отвечает с использованием локальных источников информации, объединив:

- **FastAPI** — REST API для приёма запросов;
- **Milvus** — векторная база данных для поиска по документации;
- **Faiss** — поиск по базе готовых QA-ответов (FAQ);
- **Sentence-Transformers** — генерация эмбеддингов;
- **Reranker (BGE)** — повторная переоценка top-k документов;
- **LLaMA** — локальная языковая модель для генерации ответов.

---

## Структура проекта

├── api.py # FastAPI endpoint ├── predict.py # Основная логика поиска и генерации ├── preprocessing.py # Предобработка и генерация эмбеддингов ├── retriever.py # Milvus/Faiss поисковик для документации ├── open_llama.py # Загрузка/дообучение модели LLaMA ├── requirements.txt # Зависимости ├── docker-compose.yaml # Milvus + API сервисы ├── Dockerfile # Образ FastAPI ├── data/ # Входные и выходные данные │ ├── selsup_articles.json # Документация │ ├── result.csv # Готовые вопросы/ответы (QA) │ ├── *.pkl # Эмбеддинги │ └── fine_tuned_openllama/ # Вес LLaMA ├── assets/ │ └── rag_architecture.png # Схема архитектуры └── README.md


---

## Установка и запуск

### 1. Клонируем проект и устанавливаем зависимости:
```bash
git clone https://github.com/your/repo.git
cd repo

2. Предобработка данных

Создаст эмбеддинги для Milvus и Faiss (для QA):

python preprocessing.py

3. Запуск через Docker Compose

docker-compose up --build

4. Тест запроса:

curl "http://localhost:8585/search?query=Как выставить товары на маркетплейс"

Переключение Faiss / Milvus

По умолчанию используется Milvus. Чтобы переключиться на Faiss:

    Открой retriever.py

    Найди строку:

USE_MILVUS = True

    Измени на:

USE_MILVUS = False

Архитектура

    Пользователь отправляет запрос → FastAPI

    Генерация эмбеддинга запроса

    Поиск:

        Документация → через Milvus или Faiss (топ-5)

        QA база → через Faiss (топ-1)

    Reranker определяет, что более релевантно

    Если документация — LLaMA генерирует ответ

    Если QA — возвращается готовый ответ


