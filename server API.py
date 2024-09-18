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

# Import error handling functions from utility.py
from utility import handle_openai_error, handle_internal_error

# Initialization of the server readiness flag
server_ready = False

# Flask app initialization
app = Flask(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    # Check if it's an OpenAIError
    if isinstance(e, openai.error.OpenAIError):
        return handle_openai_error(e)
    else:
        # For all other exceptions, handle as internal errors
        return handle_internal_error(e)

# Define the /status endpoint
@app.route('/status', methods=['GET'])
def status():
    if server_ready:
        return jsonify({'status': 'ready'}), 200
    else:
        return jsonify({'status': 'not_ready'}), 503

# Asynchronous function to handle incoming messages from the client
@app.route('/api/message', methods=['POST'])
async def handle_message():
    if not server_ready:
        return jsonify({'error': 'Server is not ready'}), 503

    logging.info("🚀 New request received.")
    data = request.get_json()
    if not data:
        logging.error("❌ No input data.")
        return jsonify({'error': 'No input data provided'}), 400

    user_message = data.get('message')
    if not user_message:
        logging.error("❌ No message to process.")
        return jsonify({'error': 'No message provided'}), 400

    logging.info(f"💬 User message: {user_message}")
    is_complex, subqueries = await analyze_and_decompose_query_with_llm(user_message)

    if is_complex:
        logging.info(f"🛠️ Subqueries found ({len(subqueries)}):")
        for subquery in subqueries:
            logging.info(f"  🔸 {subquery}")
        tasks = [process_subquery(subquery) for subquery in subqueries]
        try:
            assistant_responses = await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"❌ Error processing subqueries: {e}")
            return jsonify({'error': str(e)}), 500

        final_response = await aggregate_responses_with_llm(assistant_responses)
        if "Error" in final_response:
            logging.error(f"❌ Error in final response: {final_response}")
            return jsonify({'error': final_response}), 500
    else:
        if "Error" in subqueries[0]:
            logging.error(f"❌ Error in subqueries: {subqueries[0]}")
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
            logging.info(f"🎯 Final answer for simple query: {final_response}")
        except Exception as e:
            logging.error(f"❌ Error in simple query: {e}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'response': final_response})

# Function to analyze the user's message and decompose it into subqueries if necessary
async def analyze_and_decompose_query_with_llm(user_message):
    logging.info("🔎 Starting query processing.")

    # Check the length of the query
    word_count = len(user_message.strip().split())
    if word_count <= 10:
        logging.info("✅ Query recognized as simple. No decomposition required.")
        return False, [user_message]

    messages = [
        {"role": "system", "content": decomposition_system_content},
        {"role": "user",
         "content": f"Break down the following query into subqueries: '{user_message}'. The responses should be brief and in the same language."}
    ]

    try:
        response = await openai_post_request(
            messages, decomposition_model_name, decomposition_max_tokens, decomposition_temperature
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        # Convert the result into a list of subqueries
        subqueries = assistant_reply.split('\n')
        subqueries = [q.strip() for q in subqueries if q.strip()]

        # Clean subqueries from numbering
        cleaned_subqueries = []
        for q in subqueries:
            q = re.sub(r'^\d+\.\s*', '', q)  # Remove leading number and spaces
            cleaned_subqueries.append(q)

        logging.info("🔍 Decomposing query into subqueries:")
        for subquery in cleaned_subqueries:
            logging.info(f"  🔸 {subquery}")

        # Check the number of subqueries
        if len(cleaned_subqueries) <= 1:
            logging.info("✅ Query does not require decomposition.")
            return False, [user_message]

        # Check similarity of subqueries to the original query
        similarity_scores = [util.cos_sim(semantic_model.encode(user_message), semantic_model.encode(q))[0][0].item() for q in cleaned_subqueries]
        avg_similarity = sum(similarity_scores) / len(similarity_scores)

        if avg_similarity > 0.9:
            logging.info("✅ Subqueries are too similar to the original query. No decomposition required.")
            return False, [user_message]

        is_complex = True
        return is_complex, cleaned_subqueries
    except Exception as e:
        logging.error(f"❌ Error during analysis and decomposition: {e}")
        return False, [f"Error: {str(e)}"]

# Function to process a single subquery
async def process_subquery(subquery):
    logging.info(f"➡️ Processing subquery: {subquery}")

    # Get the recommended model based on semantic similarity
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
        logging.info(f"✔️ Subquery response: {assistant_reply}")
        return assistant_reply
    except Exception as e:
        logging.error(f"❌ Error processing subquery: {e}")
        return f"Error: {str(e)}"

# Function to aggregate the responses from subqueries into a final answer
async def aggregate_responses_with_llm(subquery_responses):
    logging.info("🔄 Merging subquery responses:")

    aggregation_prompt = "Here are the responses to the subqueries:\n"
    for i, response in enumerate(subquery_responses, 1):
        logging.info(f"  {i}. {response}")
        aggregation_prompt += f"Subquery {i}: {response}\n"

    aggregation_prompt += "Combine these responses into a short, coherent, and clear final answer in the same language as the original question."

    messages = [
        {"role": "system", "content": aggregation_system_content},
        {"role": "user", "content": aggregation_prompt}
    ]

    try:
        response = await openai_post_request(
            messages, aggregation_model_name, aggregation_max_tokens, aggregation_temperature
        )
        aggregated_response = response['choices'][0]['message']['content'].strip()

        logging.info(f"🎯 Final answer: {aggregated_response}")
        return aggregated_response
    except Exception as e:
        logging.error(f"❌ Error during aggregation: {e}")
        return f"Error: {str(e)}"

# Function to send a request to the OpenAI API
async def openai_post_request(messages, model_name, max_tokens, temperature):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",  # API key obtained from config.ini
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3  # Number of attempts
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Check for successful request
            return response.json()  # Return JSON response from OpenAI API
        except httpx.ReadTimeout:
            logging.warning(f"⏳ Timeout on attempt {attempt + 1} when calling OpenAI API.")
            if attempt == retries - 1:
                raise
        except httpx.HTTPStatusError as e:
            logging.error(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
            raise

# Function to load model recommendations from the configuration file
def load_model_recommendations():
    config_models = configparser.ConfigParser(interpolation=None)
    # Read the file with UTF-8 encoding
    with open('models_config.ini', 'r', encoding='utf-8') as f:
        config_models.read_file(f)

    model_recommendations = {}
    available_topics = []

    for topic, model in config_models['models'].items():
        available_topics.append(topic)
        model_recommendations[topic] = model

    return available_topics, model_recommendations

# Function to recommend a language model based on semantic similarity
def recommend_llm_by_semantics(user_message):
    available_topics, model_recommendations = load_model_recommendations()

    # Generate embeddings for topics
    topic_embeddings = semantic_model.encode(available_topics, convert_to_tensor=True)

    # Embedding of the user's message
    message_embedding = semantic_model.encode(user_message, convert_to_tensor=True)

    # Calculate cosine similarity
    similarities = util.cos_sim(message_embedding, topic_embeddings)[0]

    # Find the topic with the highest similarity
    best_match_idx = similarities.argmax()
    best_match_topic = available_topics[best_match_idx]
    best_match_score = similarities[best_match_idx].item()

    # Select the model corresponding to the found topic
    selected_model = model_recommendations.get(best_match_topic, model_recommendations.get('Default', 'gpt-3.5-turbo'))

    # Log information
    logging.info(f"🤖 Selected model for topic '{best_match_topic}' (similarity {best_match_score:.2f}): {selected_model}")

    return selected_model

if __name__ == '__main__':
    # Read the configuration file with interpolation disabled
    config = configparser.ConfigParser(interpolation=None)
    with open('config.ini', 'r', encoding='utf-8') as configfile:
        config.read_file(configfile)

    # **Logging settings**
    log_level_str = config['logging'].get('level', 'INFO')
    log_format = config['logging'].get('format', '%(asctime)s [%(levelname)s] %(message)s')

    # Convert log level string to logging level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    # **Logging configuration**
    LOG_LEVEL = log_level
    LOGFORMAT = "%(log_color)s" + log_format + "%(reset)s"

    # Color scheme for log levels
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

    logging.root.setLevel(LOG_LEVEL)
    logging.root.handlers = [handler]  # Replace existing handlers with the new one
    logging.getLogger("httpx").setLevel(logging.WARNING)

    logging.info("🔧 Initializing server and loading models...")

    # OpenAI API key
    api_key = config['openai'].get('api_key')
    if not api_key:
        logging.error("OpenAI API key is not set. Please set it in the 'config.ini' file.")
        exit(1)

    openai.api_key = api_key  # Set the OpenAI API key

    # Decomposition model settings
    decomposition_model_name = config['decomposition_model']['name']
    decomposition_max_tokens = config['decomposition_model'].getint('max_tokens')
    decomposition_temperature = config['decomposition_model'].getfloat('temperature')

    # Aggregation model settings
    aggregation_model_name = config['aggregation_model']['name']
    aggregation_max_tokens = config['aggregation_model'].getint('max_tokens')
    aggregation_temperature = config['aggregation_model'].getfloat('temperature')

    # System prompts
    default_system_content = config['prompt']['default_system_content']
    decomposition_system_content = config['prompt']['decomposition_system_content']
    aggregation_system_content = config['prompt']['aggregation_system_content']

    # Semantic search settings
    similarity_threshold = config['semantic_search'].getfloat('similarity_threshold', fallback=0.5)
    embedding_model_name = config['semantic_search']['embedding_model_name']

    # Server settings
    server_host = config['server'].get('host', '127.0.0.1')
    server_port = config['server'].getint('port', 5000)

    # Initialize the semantic model
    logging.info(f"📥 Loading semantic model '{embedding_model_name}'...")
    semantic_model = SentenceTransformer(embedding_model_name)
    logging.info(f"✅ Semantic model '{embedding_model_name}' loaded.")

    # Set the server readiness flag
    server_ready = True
    logging.info("🚀 Server is ready.")

    # Start the Flask app with host and port from the config
    app.run(host=server_host, port=server_port, debug=False)
