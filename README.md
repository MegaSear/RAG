# RAG
# Tourist RAG — домашнее задание по RAG

Построение Retrieval-Augmented Generation системы для ответов на вопросы о туристических достопримечательностях: очистка мультимодального датасета, векторный поиск на FAISS с reranking'ом, генерация ответов LLM и оценка качества по метрикам в духе RAGAS.

> Ноутбук выполнен в рамках домашнего задания «Школы глубокого обучения ФПМИ МФТИ».

## Датасет

Каждая запись содержит:

- название достопримечательности (`Name`) и город (`City`);
- идентификатор WikiData (`WikiData`);
- координаты (`Lat`, `Lon`);
- текстовое описание, извлечённое из WikiData (`description`);
- изображение в base64 (`image`);
- сгенерированное BLIP-описание изображения (`en_txt`).

Один и тот же объект может встречаться несколько раз с разными (не всегда качественными) изображениями — текстовые описания компенсируют это.

## Пайплайн

1. **Загрузка и очистка данных**
   - Удаление пустых/некорректных строк.
   - Формирование единого текстового представления (`Name` + `City` + `description` + `en_txt`).
   - Детекция выбросов (нетуристические записи, мемы, баннеры и т.п.) через эмбеддинги + `IsolationForest`.
   - Дедупликация: для каждой пары `(City, Name)` выбирается лучшая запись (по суммарной длине текста).

2. **Построение RAG**
   - Чанкинг через `RecursiveCharacterTextSplitter` (в данном случае — почти no-op, тексты короткие и уже разбиты по смыслу).
   - Эмбеддинги — `intfloat/multilingual-e5-base` (`HuggingFaceEmbeddings`).
   - Векторная база — `FAISS` (`langchain_community.vectorstores`).
   - Reranking — `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (вместо `ragatouille` — из-за проблем с установкой зависимостей).
   - Генерация ответа — `Qwen/Qwen2.5-3B-Instruct`.

3. **Визуализация эмбеддингов**
   - PCA и UMAP для 2D-проекции эмбеддингов чанков.

4. **Оценка качества (RAGAS-style, реализовано вручную)**
   - `answer_relevancy` — косинусное сходство между эмбеддингом исходного вопроса и эмбеддингами вопросов, сгенерированных LLM по ответу.
   - `faithfulness` — доля утверждений из ответа, подтверждаемых контекстом (LLM-as-judge).
   - `context_recall` — доля утверждений из ground truth, покрытых найденным контекстом.
   - `context_precision` — average precision@k по релевантности чанков контекста (доп. задание).

## Требования

- Python 3.10+
- GPU с CUDA (генерация и reranking выполняются на `cuda`)
- Основные библиотеки:
  ```
  sentence-transformers
  faiss-cpu
  langchain, langchain-community, langchain-core, langchain-huggingface, langchain-text-splitters
  huggingface_hub
  transformers, torch
  pandas, numpy, scikit-learn
  matplotlib, plotly
  umap-learn
  gdown
  ```

## Данные

По умолчанию ноутбук ожидает CSV по пути `/kaggle/input/datasets/vasilijkrukovskij/wikitexts/data.csv` (Kaggle-окружение).
Для запуска вне Kaggle установите `drive = True` в соответствующей ячейке — данные будут скачаны с Google Drive через `gdown`.

## Запуск

1. Установить зависимости (см. ячейку с `pip install`).
2. Выполнить ячейки последовательно сверху вниз.
3. Пример запроса к RAG:
   ```python
   query = "Where is the Museum of Dolls and Children's Books 'Land of Wonders' located?"
   answer, sources = rag(query, reranker, vectorstore, gen_tokenizer, gen_model,
                          top_k=25, top_n=5, threshold_p=0.7)
   ```

## Известные ограничения
- Оценка метрик (`evaluate_rag`) делает отдельный LLM-вызов на каждый claim/контекст и может быть медленной на большом числе сэмплов.
