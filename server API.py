import httpx
import logging
from logging import StreamHandler
from colorlog import ColoredFormatter
from flask import Flask, request, jsonify, Response
import asyncio
import configparser
import traceback
from sentence_transformers import SentenceTransformer, util
import re

# Инициализация флага готовности сервера
server_ready = False

# Настройка логирования
LOG_LEVEL = logging.INFO
LOGFORMAT = "%(log_color)s%(asctime)s [%(levelname)s] %(message)s%(reset)s"

# Цветовая схема
COLOR_SCHEME = {
    'DEBUG':    'cyan',
    'INFO':     'green',
    'WARNING':  'yellow',
    'ERROR':    'red',
    'CRITICAL': 'bold_red',
}

formatter = ColoredFormatter(
    LOGFORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    reset=True,
    log_colors=COLOR_SCHEME
)

handler = StreamHandler()
handler.setFormatter(formatter)

logging.root.setLevel(LOG_LEVEL)
logging.root.addHandler(handler)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Flask app initialization
app = Flask(__name__)

# Определение эндпоинта /status
@app.route('/status', methods=['GET'])
def status():
    if server_ready:
        return jsonify({'status': 'ready'}), 200
    else:
        return jsonify({'status': 'not_ready'}), 503

# Асинхронная функция для обработки сообщения
@app.route('/api/message', methods=['POST'])
async def handle_message():
    if not server_ready:
        return jsonify({'error': 'Server is not ready'}), 503

    logging.info("🚀 Получен новый запрос.")
    data = request.get_json()
    if not data:
        logging.error("❌ Нет входных данных.")
        return jsonify({'error': 'No input data provided'}), 400

    user_message = data.get('message')
    if not user_message:
        logging.error("❌ Нет сообщения для обработки.")
        return jsonify({'error': 'No message provided'}), 400

    logging.info(f"💬 Сообщение пользователя: {user_message}")
    is_complex, subqueries = await analyze_and_decompose_query_with_llm(user_message)

    if is_complex:
        logging.info(f"🛠️ Найдены подзапросы ({len(subqueries)}):")
        for subquery in subqueries:
            logging.info(f"  🔸 {subquery}")
        tasks = [process_subquery(subquery) for subquery in subqueries]
        try:
            assistant_responses = await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"❌ Ошибка при обработке подзапросов: {e}")
            return jsonify({'error': str(e)}), 500

        final_response = await aggregate_responses_with_llm(assistant_responses)
        if "Error" in final_response:
            logging.error(f"❌ Ошибка в финальном ответе: {final_response}")
            return jsonify({'error': final_response}), 500
    else:
        if "Error" in subqueries[0]:
            logging.error(f"❌ Ошибка в подзапросах: {subqueries[0]}")
            return jsonify({'error': subqueries[0]}), 500

        messages = [
            {"role": "system", "content": default_system_content},
            {"role": "user", "content": user_message}
        ]
        try:
            response = await openai_post_request(
                messages, decomposition_model_name, decomposition_max_tokens, decomposition_temperature
            )
            final_response = response['choices'][0]['message']['content'].strip()
            logging.info(f"🎯 Финальный ответ для простого запроса: {final_response}")
        except Exception as e:
            logging.error(f"❌ Ошибка в простом запросе: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'response': final_response})

# Функция для анализа и декомпозиции запроса
async def analyze_and_decompose_query_with_llm(user_message):
    logging.info("🔎 Начало обработки запроса.")

    # Проверка длины запроса
    word_count = len(user_message.strip().split())
    if word_count <= 10:
        logging.info("✅ Запрос распознан как простой. Не требуется разбиение.")
        return False, [user_message]

    messages = [
        {"role": "system", "content": decomposition_system_content},
        {"role": "user",
         "content": f"Разбей данный запрос на подзапросы: '{user_message}'. Ответы должны быть краткими и на том же языке."}
    ]

    try:
        response = await openai_post_request(
            messages, decomposition_model_name, decomposition_max_tokens, decomposition_temperature
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        # Преобразуем результат в список подзапросов
        subqueries = assistant_reply.split('\n')
        subqueries = [q.strip() for q in subqueries if q.strip()]

        # Очистка подзапросов от номеров
        cleaned_subqueries = []
        for q in subqueries:
            q = re.sub(r'^\d+\.\s*', '', q)  # Удаляем номер в начале строки
            cleaned_subqueries.append(q)

        logging.info("🔍 Разбиение запроса на подзапросы:")
        for subquery in cleaned_subqueries:
            logging.info(f"  🔸 {subquery}")

        # Проверка количества подзапросов
        if len(cleaned_subqueries) <= 1:
            logging.info("✅ Запрос не требует разбиения.")
            return False, [user_message]

        # Проверка на схожесть подзапросов с исходным запросом
        similarity_scores = [util.cos_sim(semantic_model.encode(user_message), semantic_model.encode(q))[0][0].item() for q in cleaned_subqueries]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        if avg_similarity > 0.9:
            logging.info("✅ Подзапросы слишком похожи на исходный запрос. Не требуется разбиение.")
            return False, [user_message]

        is_complex = True
        return is_complex, cleaned_subqueries
    except Exception as e:
        logging.error(f"❌ Ошибка при анализе и декомпозиции: {e}")
        return False, [f"Error: {str(e)}"]

# Функция для обработки подзапроса
async def process_subquery(subquery):
    logging.info(f"➡️ Обработка подзапроса: {subquery}")

    # Получаем рекомендованную модель
    recommended_model = recommend_llm_by_semantics(subquery)

    messages = [
        {"role": "system", "content": default_system_content},
        {"role": "user", "content": subquery}
    ]

    try:
        response = await openai_post_request(
            messages, recommended_model, decomposition_max_tokens, decomposition_temperature
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        logging.info(f"✔️ Ответ на подзапрос: {assistant_reply}")
        return assistant_reply
    except Exception as e:
        logging.error(f"❌ Ошибка обработки подзапроса: {e}")
        return f"Error: {str(e)}"

# Функция для агрегации ответов
async def aggregate_responses_with_llm(subquery_responses):
    logging.info("🔄 Слияние ответов подзапросов:")

    aggregation_prompt = "Вот ответы на подзапросы:\n"
    for i, response in enumerate(subquery_responses, 1):
        logging.info(f"  {i}. {response}")
        aggregation_prompt += f"Подзапрос {i}: {response}\n"

    aggregation_prompt += "Собери эти ответы в короткий, связный и четкий финальный ответ на том же языке, на котором был исходный вопрос."

    messages = [
        {"role": "system", "content": aggregation_system_content},
        {"role": "user", "content": aggregation_prompt}
    ]

    try:
        response = await openai_post_request(
            messages, aggregation_model_name, aggregation_max_tokens, aggregation_temperature
        )
        aggregated_response = response['choices'][0]['message']['content'].strip()

        logging.info(f"🎯 Финальный ответ: {aggregated_response}")
        return aggregated_response
    except Exception as e:
        logging.error(f"❌ Ошибка при агрегации: {e}")
        return f"Error: {str(e)}"

# Функция для отправки запроса в OpenAI API
async def openai_post_request(messages, model_name, max_tokens, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",  # API ключ, полученный из config.ini
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3  # Количество попыток
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Проверяем успешность запроса
            return response.json()  # Возвращаем JSON-ответ от OpenAI API
        except httpx.ReadTimeout:
            logging.warning(f"⏳ Таймаут при попытке {attempt + 1} обращения к OpenAI API.")
            if attempt == retries - 1:
                raise
        except httpx.HTTPStatusError as e:
            logging.error(f"❌ HTTP ошибка: {e.response.status_code} - {e.response.text}")
            raise

# Функция для загрузки рекомендаций моделей
def load_model_recommendations():
    config = configparser.ConfigParser()
    # Чтение файла с указанием кодировки UTF-8
    with open('models_config.ini', 'r', encoding='utf-8') as f:
        config.read_file(f)

    model_recommendations = {}
    available_topics = []

    for topic, model in config['models'].items():
        available_topics.append(topic)
        model_recommendations[topic] = model

    return available_topics, model_recommendations

# Функция для рекомендации модели на основе семантического сходства
def recommend_llm_by_semantics(user_message):
    available_topics, model_recommendations = load_model_recommendations()

    # Генерируем эмбеддинги для тем
    topic_embeddings = semantic_model.encode(available_topics, convert_to_tensor=True)

    # Эмбеддинг пользовательского сообщения
    message_embedding = semantic_model.encode(user_message, convert_to_tensor=True)

    # Вычисляем косинусное сходство
    similarities = util.cos_sim(message_embedding, topic_embeddings)[0]

    # Находим тему с максимальным сходством
    best_match_idx = similarities.argmax()
    best_match_topic = available_topics[best_match_idx]
    best_match_score = similarities[best_match_idx].item()

    # Выбираем модель, соответствующую найденной теме
    selected_model = model_recommendations.get(best_match_topic, model_recommendations.get('Default', 'gpt-3.5-turbo'))

    # Логируем информацию
    logging.info(f"🤖 Выбранная модель для темы '{best_match_topic}' (сходство {best_match_score:.2f}): {selected_model}")

    return selected_model

if __name__ == '__main__':
    logging.info("🔧 Инициализация сервера и загрузка моделей...")

    # Чтение конфигурационного файла
    config = configparser.ConfigParser()
    with open('config.ini', 'r', encoding='utf-8') as configfile:
        config.read_file(configfile)

    # Установка OpenAI API ключа
    api_key = config['openai']['api_key']

    # Загрузка настроек моделей
    decomposition_model_name = config['decomposition_model']['name']
    decomposition_max_tokens = config['decomposition_model'].getint('max_tokens')
    decomposition_temperature = config['decomposition_model'].getfloat('temperature')

    aggregation_model_name = config['aggregation_model']['name']
    aggregation_max_tokens = config['aggregation_model'].getint('max_tokens')
    aggregation_temperature = config['aggregation_model'].getfloat('temperature')

    # Загрузка системных промптов
    default_system_content = config['prompt']['default_system_content']
    decomposition_system_content = config['prompt']['decomposition_system_content']
    aggregation_system_content = config['prompt']['aggregation_system_content']

    # Загрузка настроек семантического поиска
    similarity_threshold = config['semantic_search'].getfloat('similarity_threshold', fallback=0.5)
    embedding_model_name = config['semantic_search']['embedding_model_name']

    # Инициализация семантической модели
    logging.info(f"📥 Загрузка семантической модели '{embedding_model_name}'...")
    semantic_model = SentenceTransformer(embedding_model_name)
    logging.info(f"✅ Семантическая модель '{embedding_model_name}' загружена.")

    # Установка флага готовности сервера
    server_ready = True
    logging.info("🚀 Сервер готов к работе.")

    app.run(debug=False)
