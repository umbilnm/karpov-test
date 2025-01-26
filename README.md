# Проект "Ассистент по выбору курсов Karpov.Courses"


## Steps

Представьте, что вы — разработчик в компании karpov.courses, которая хочет создать интеллектуального помощника для студентов. Этот помощник должен рекомендовать курсы для студентов на основе их интересов и целей. Ваша задача — реализовать этот проект, следуя шагам, описанным ниже.


### Шаг 1: Парсинг страниц курсов

На первом этапе мы будем собирать ссылки на курсы с сайта Karpov.Courses. 

Напишите функцию `parse_courses()`, которая возвращает список URL курсов, находящихся на лендинге karpov.courses.

После чего нам необходимо заглянуть на страницу каждого курса, получить весь текст и сохранить в файл `data/full_texts.json`, где ключом будет ссылка на курс, а значением текст страницы.

Для выполнения потребуется использовать библиотеку `requests` для получения HTML-кода страниц и библиотеку `BeautifulSoup` для извлечения текста.

**Функции:**
- `parse_courses()`: возвращает список URL курсов.
- `extract_pages(hrefs: List[str])`: извлекает текст со страниц курсов и сохраняет в `data/full_texts.json`.

**Полезные ссылки:**
- [Requests Documentation](https://docs.python-requests.org/en/latest/): Официальная документация для работы с HTTP-запросами.
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): Официальная документация для работы с HTML и XML.

### Шаг 2: ~~Chunking~~ Summarization

Отлично, теперь у нас есть текст-описание каждого курса, но если посмотреть хотя бы на один из них, то можно увидеть, что текст очень большой и содержит много ненужной информации, в том числе упоминание других курсов, различные ссылки, контакты и т.д. 
Если мы попытаемся разбить текст каждой страницы на чанки, то некоторые части будут полностью состоять из нерелевантной информации. Если же мы попытаемся брать эмбеддинг каждого курса целиком, то эмбеддинг будет нерепрезентативным, так как он будет зашумлен.

Есть два пути решения этой проблемы:
1. Очищение текста от ненужной информации.(долго и необязательно точно)
2. Использовать API OpenAI для создания кратких описаний курсов.

Так как у нас всего 21 ссылка курсов, то использование LLM для суммаризации будет очень эффективным. Для каждого курса за один вызов OpenAI API мы получим краткое описание курса, эмбеддинг которого и будем использовать в дальнейшем.

Но это не значит что всегда следует использовать суммаризацию. Это лишь один из приемов. Выбор метода зависит от конкретной задачи и данных.

Сначала вам необходимо написать функцию summarize_text, которая принимает текст курса и возвращает краткое описание курса полученное с помощью LLM.
После чего вам необходимо написать функцию summarize_pages, которая обрабатывает все тексты и сохраняет результаты в `data/summaries.json`, где ключом будет ссылка на курс, а значением краткое описание.


**Функции:**
- `summarize_text(text_to_summarize: str)`: возвращает краткое описание текста.
- `summarize_pages()`: обрабатывает все тексты курсов и сохраняет результаты в `data/summaries.json`.


### Шаг 3: Создание векторного индекса

В качестве эмбеддинга мы будем использовать `text-embedding-ada-002` из OpenAI.
Используя эмбеддинги, мы можем эффективно сравнивать текстовые данные. Например, эмбеддинг для каждого курса будет содержать информацию о его содержании и темах. Это позволяет нам находить курсы, которые наиболее близки к пользовательскому запросу, сравнивая эмбеддинги запроса и курсов. Чем ближе эмбеддинги друг к другу в пространстве, тем более схожи их семантические значения.

Первым делом нужно написать функцию `embed_text`, которая принимает текст и возвращает его векторное представление.

Теперь мы создадим векторный индекс для быстрого поиска похожих курсов. Мы используем библиотеку `faiss` для работы с векторными базами данных. FAISS (Facebook AI Similarity Search) — это библиотека, разработанная Facebook AI Research, предназначенная для эффективного поиска по векторным базам данных. Она позволяет быстро находить похожие элементы на основе их векторных представлений. 

В функции `create_index` мы создадим faiss индекс из векторных представлений текстов. Функция должна возвращать (index: faiss.Index, urls: List[str], embeddings: np.ndarray), где url на i-ой позиции соответствует вектору на i-ой позиции.

Далее в функции `save_index` сохраняем faiss index и  создадим вспомогательную структуру, которая будет хранить метаинформацию о курсах. Эта структура данных будет включать URL курса, краткое описание (summary) и векторное представление текста (embedding). Она необходима для упрощения доступа к данным на retrival этапе.

_Так как у нас всего 21 документ, мы могли бы обойтись обычным поиском соседей без оптимизаций, faiss здесь скорее для знакомства с векторными базами данных._

**Функции:**
- `embed_text(text: str)`: возвращает векторное представление текста, используя OpenAI API.
- `create_index(summaries: Dict[str, str])`: создает индекс из векторных представлений текстов.
- `save_index(...)`: сохраняет индекс и данные в файлы.

**Полезные ссылки:**
- [FAISS Documentation](https://faiss.ai/): Официальная документация FAISS для работы с векторными базами данных.
- [Understanding Vector Databases](https://towardsdatascience.com/understanding-vector-databases-1f1f8b1c1c3b): Статья о том, как работают векторные базы данных и их применение.

### Шаг 4: Поиск похожих курсов

На этом этапе мы реализуем функцию поиска похожих курсов на основе пользовательского запроса. Мы будем использовать созданный ранее векторный индекс. Поиск происходит путем сравнения векторного представления запроса с векторами курсов, что позволяет быстро находить наиболее релевантные результаты. Функция `retrieve_similar` возвращает список кортежей (url, summary)  из top_k наиболее похожих курсов на основе запроса пользователя. top_k -- гиперпараметр, который можно настроить, по умолчанию 3.


**Функции:**
- `load_index(index_path: str, documents_path: str)`: загружает FAISS индекс и документы из указанных файлов.
- `retrieve_similar(query: str, top_k: int = 3)`: возвращает список из top_k наиболее похожих курсов на основе запроса пользователя.

**Полезные ссылки:**
- [Nearest Neighbor Search](https://en.wikipedia.org/wiki/Nearest_neighbor_search): Введение в алгоритмы поиска ближайших соседей.
- [FAISS Index Search](https://github.com/facebookresearch/faiss/wiki/Indexing-1M-vectors): Пример использования FAISS для поиска в индексах.

### Шаг 5: Веб-интерфейс с использованием Streamlit

На последнем этапе мы создадим веб-интерфейс, который позволит пользователям взаимодействовать с нашим ассистентом. Мы используем библиотеку `Streamlit` для создания простого и удобного интерфейса. 


Как происходит обработка запроса пользователя:
1) Пользователь вводит запрос в поле для ввода.
2) Ищем top_k ближайших соседей для пользовательского запроса в векторном индексе, используя функцию `retrieve_similar`.
3) Кладем информацию о найденных курсах context при вызове LLM

Как может выглядеть промпт:

```
You are a helpful assistant that helps users find courses on Karpov.Courses.

User query: {user_query}

Context: {context}

Your task is to find the most relevant courses for the user query.
```

Но это лишь пример, вам нужно будет написать свой промпт, где вы можете применить любые свои идеи!
4) Показываем пользователю ответ LLM.

**Функции:**
- `get_course_recommendation(user_query: str)`: возвращает рекомендации по курсам на основе запроса пользователя.
- `main()`: запускает веб-приложение.
