import httpx
import logging
from logging import StreamHandler
from colorlog import ColoredFormatter
from flask import Flask, request, jsonify, Response
import asyncio
import configparser
from sentence_transformers import SentenceTransformer, util
import re
import os
import openai
from dotenv import load_dotenv

# Import error handling functions from utility.py
from utility import handle_openai_error, handle_internal_error, handle_request_error

# Loading environment variables, like API keys
load_dotenv()

# Flask app initialization
app = Flask(__name__)

# Initialization of the server readiness flag
server_ready = False

# Global variables for models and configurations
config = None
semantic_model = None


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, openai.error.OpenAIError):
        return handle_openai_error(e)
    else:
        return handle_internal_error(e)


@app.route('/status', methods=['GET'])
def status():
    if server_ready:
        return jsonify({'status': 'ready'}), 200
    else:
        return jsonify({'status': 'not_ready'}), 503


@app.route('/api/message', methods=['POST'])
async def handle_message():
    if not server_ready:
        return jsonify({'error': 'Server is not ready'}), 503

    logging.info("🚀 New request received.")
    data = request.get_json()
    if not data or 'message' not in data:
        return handle_request_error("No message provided")

    user_message = data['message']
    logging.info(f"💬 User message: {user_message}")

    try:
        final_response = await process_message(user_message)
        return jsonify({'response': final_response})
    except Exception as e:
        return handle_internal_error(e)


async def process_message(user_message):
    is_complex, subqueries = await analyze_and_decompose_query_with_llm(user_message)

    if is_complex:
        logging.info(f"🛠️ Subqueries found ({len(subqueries)}):")
        for subquery in subqueries:
            logging.info(f"  🔸 {subquery}")

        try:
            assistant_responses = await asyncio.gather(*[process_subquery(subquery) for subquery in subqueries])
            final_response = await aggregate_responses_with_llm(assistant_responses)
        except Exception as e:
            logging.error(f"❌ Error processing subqueries: {e}")
            raise
    else:
        final_response = await process_simple_query(user_message)

    return final_response


async def analyze_and_decompose_query_with_llm(user_message):
    logging.info("🔎 Starting query processing.")

    word_count = len(user_message.strip().split())
    if word_count <= 10:
        logging.info("✅ Query recognized as simple. No decomposition required.")
        return False, [user_message]

    messages = [
        {"role": "system", "content": config['prompt']['decomposition_system_content']},
        {"role": "user",
         "content": f"Break down the following query into subqueries: '{user_message}'. The responses should be brief and in the same language."}
    ]

    try:
        response = await openai_post_request(
            messages,
            config['decomposition_model']['name'],
            int(config['decomposition_model']['max_tokens']),
            float(config['decomposition_model']['temperature'])
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        subqueries = [re.sub(r'^\d+\.\s*', '', q.strip()) for q in assistant_reply.split('\n') if q.strip()]

        logging.info("🔍 Decomposing query into subqueries:")
        for subquery in subqueries:
            logging.info(f"  🔸 {subquery}")

        if len(subqueries) <= 1:
            logging.info("✅ Query does not require decomposition.")
            return False, [user_message]

        similarity_scores = [util.cos_sim(semantic_model.encode(user_message), semantic_model.encode(q))[0][0].item()
                             for q in subqueries]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        if avg_similarity > 0.9:
            logging.info("✅ Subqueries are too similar to the original query. No decomposition required.")
            return False, [user_message]

        return True, subqueries
    except Exception as e:
        logging.error(f"❌ Error during analysis and decomposition: {e}")
        raise


async def process_subquery(subquery):
    logging.info(f"➡️ Processing subquery: {subquery}")

    recommended_model = recommend_llm_by_semantics(subquery)

    messages = [
        {"role": "system", "content": config['prompt']['default_system_content']},
        {"role": "user", "content": subquery}
    ]

    try:
        response = await openai_post_request(
            messages,
            recommended_model,
            int(config['decomposition_model']['max_tokens']),
            float(config['decomposition_model']['temperature'])
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        logging.info(f"✔️ Subquery response: {assistant_reply}")
        return assistant_reply
    except Exception as e:
        logging.error(f"❌ Error processing subquery: {e}")
        raise


async def aggregate_responses_with_llm(subquery_responses):
    logging.info("🔄 Merging subquery responses:")

    aggregation_prompt = "Here are the responses to the subqueries:\n"
    for i, response in enumerate(subquery_responses, 1):
        logging.info(f"  {i}. {response}")
        aggregation_prompt += f"Subquery {i}: {response}\n"

    aggregation_prompt += "Combine these responses into a short, coherent, and clear final answer in the same language as the original question."

    messages = [
        {"role": "system", "content": config['prompt']['aggregation_system_content']},
        {"role": "user", "content": aggregation_prompt}
    ]

    try:
        response = await openai_post_request(
            messages,
            config['aggregation_model']['name'],
            int(config['aggregation_model']['max_tokens']),
            float(config['aggregation_model']['temperature'])
        )
        aggregated_response = response['choices'][0]['message']['content'].strip()

        logging.info(f"🎯 Final answer: {aggregated_response}")
        return aggregated_response
    except Exception as e:
        logging.error(f"❌ Error during aggregation: {e}")
        raise


async def process_simple_query(user_message):
    messages = [
        {"role": "system", "content": config['prompt']['default_system_content']},
        {"role": "user", "content": user_message}
    ]
    try:
        response = await openai_post_request(
            messages,
            config['decomposition_model']['name'],
            int(config['decomposition_model']['max_tokens']),
            float(config['decomposition_model']['temperature'])
        )
        final_response = response['choices'][0]['message']['content'].strip()
        logging.info(f"🎯 Final answer for simple query: {final_response}")
        return final_response
    except Exception as e:
        logging.error(f"❌ Error in simple query: {e}")
        raise


async def openai_post_request(messages, model_name, max_tokens, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3 # may be set to a config variable
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            logging.warning(f"⏳ Timeout on attempt {attempt + 1} when calling OpenAI API.")
            if attempt == retries - 1:
                raise
        except httpx.HTTPStatusError as e:
            logging.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise


def load_model_recommendations():
    config_models = configparser.ConfigParser(interpolation=None)
    with open('models_config.ini', 'r', encoding='utf-8') as f:
        config_models.read_file(f)

    model_recommendations = {}
    available_topics = []

    for topic, model in config_models['models'].items():
        available_topics.append(topic)
        model_recommendations[topic] = model

    return available_topics, model_recommendations


def recommend_llm_by_semantics(user_message):
    """ This is a placeholder for future recommender system """
    available_topics, model_recommendations = load_model_recommendations()

    topic_embeddings = semantic_model.encode(available_topics, convert_to_tensor=True)
    message_embedding = semantic_model.encode(user_message, convert_to_tensor=True)

    similarities = util.cos_sim(message_embedding, topic_embeddings)[0]

    best_match_idx = similarities.argmax()
    best_match_topic = available_topics[best_match_idx]
    best_match_score = similarities[best_match_idx].item()

    selected_model = model_recommendations.get(best_match_topic, model_recommendations.get('Default', 'gpt-3.5-turbo'))

    logging.info(
        f"🤖 Selected model for topic '{best_match_topic}' (similarity {best_match_score:.2f}): {selected_model}")

    return selected_model


def load_configuration():
    config = configparser.ConfigParser(interpolation=None)
    with open('config.ini', 'r', encoding='utf-8') as configfile:
        config.read_file(configfile)
    return config


def setup_logging(config):
    log_level_str = config['logging'].get('level', 'INFO')
    log_format = config['logging'].get('format', '%(asctime)s [%(levelname)s] %(message)s')

    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    LOGFORMAT = "%(log_color)s" + log_format + "%(reset)s"

    COLOR_SCHEME = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
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

    logging.root.setLevel(log_level)
    logging.root.handlers = [handler]
    logging.getLogger("httpx").setLevel(logging.WARNING)


def initialize_server(config):
    global server_ready, semantic_model

    logging.info("🔧 Initializing server and loading models...")

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        logging.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    embedding_model_name = os.getenv('EMBEDDING_MODEL', config['semantic_search']['embedding_model_name'])
    logging.info(f"📥 Loading semantic model '{embedding_model_name}'...")
    try:
        semantic_model = SentenceTransformer(embedding_model_name)
        logging.info(f"✅ Semantic model '{embedding_model_name}' loaded.")
    except Exception as e:
        logging.error(f"❌ Failed to load semantic model: {str(e)}")
        exit(1)

    server_ready = True
    logging.info("🚀 Server is ready.")


if __name__ == '__main__':
    config = load_configuration()
    setup_logging(config)
    initialize_server(config)

    server_host = os.getenv('SERVER_HOST', config['server'].get('host', '127.0.0.1'))
    server_port = os.getenv('SERVER_PORT', config['server'].get('port', '5000'))

    app.run(host=server_host, port=server_port, debug=False)