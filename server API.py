import httpx
import logging
from flask import Flask, request, jsonify
import asyncio
import configparser
import traceback

# Настроим логирование
logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read('config.ini')

# Set OpenAI API key
api_key = config['openai']['api_key']

# Load decomposition model settings
decomposition_model_name = config['decomposition_model']['name']
decomposition_max_tokens = config['decomposition_model'].getint('max_tokens')
decomposition_temperature = config['decomposition_model'].getfloat('temperature')

# Load aggregation model settings
aggregation_model_name = config['aggregation_model']['name']
aggregation_max_tokens = config['aggregation_model'].getint('max_tokens')
aggregation_temperature = config['aggregation_model'].getfloat('temperature')

# Load system prompts
default_system_content = config['prompt']['default_system_content']
decomposition_system_content = config['prompt']['decomposition_system_content']
aggregation_system_content = config['prompt']['aggregation_system_content']

# Initialize the Flask app
app = Flask(__name__)

# Register error handlers
@app.errorhandler(500)
def handle_internal_error(error):
    logging.error(f"Internal Server Error: {error}")
    return jsonify({'error': 'Internal Server Error'}), 500

async def openai_post_request(messages, model_name, max_tokens, temperature):
    """
    Асинхронно отправляет POST-запрос в OpenAI API с использованием httpx с таймаутом и повторными попытками.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    retries = 3  # Количество повторных попыток
    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:  # Увеличен таймаут
                response = await client.post(url, json=payload, headers=headers)

            response.raise_for_status()  # Поднимет исключение, если статус не 200
            return response.json()
        except httpx.ReadTimeout:
            logging.warning(f"Timeout occurred on attempt {attempt + 1} for OpenAI API.")
            if attempt == retries - 1:
                raise
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise

async def analyze_and_decompose_query_with_llm(user_message):
    logging.info(f"Analyzing and decomposing query: {user_message}")
    messages = [
        {"role": "system", "content": decomposition_system_content},
        {"role": "user", "content": f"Analyze the following query. '{user_message}' If it is complex, break it down into smaller subqueries."}
    ]

    try:
        response = await openai_post_request(
            messages, decomposition_model_name, decomposition_max_tokens, decomposition_temperature
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()

        # Преобразуем результат в список подзапросов
        subqueries = assistant_reply.split('\n')
        subqueries = [q.strip() for q in subqueries if q.strip()]

        logging.info(f"Subqueries generated: {subqueries}")
        is_complex = len(subqueries) > 1
        return is_complex, subqueries
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error: {str(e)} - {e.response.text}")  # Логируем полный текст ответа сервера
        return False, [f"Error: {str(e)}"]
    except Exception as e:
        logging.error(f"General error during analysis: {str(e)}")
        logging.error(traceback.format_exc())  # Добавляем трейсбек для большей информативности
        return False, [f"Error: {str(e)}"]

async def process_subquery(subquery):
    logging.info(f"Processing subquery: {subquery}")
    messages = [
        {"role": "system", "content": default_system_content},
        {"role": "user", "content": subquery}
    ]
    try:
        response = await openai_post_request(
            messages, decomposition_model_name, decomposition_max_tokens, decomposition_temperature
        )
        assistant_reply = response['choices'][0]['message']['content'].strip()
        logging.info(f"Response for subquery '{subquery}': {assistant_reply}")
        return assistant_reply
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error processing subquery: {str(e)} - {e.response.text}")
        return f"Error: {str(e)}"
    except Exception as e:
        logging.error(f"Error processing subquery: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)}"

async def aggregate_responses_with_llm(subquery_responses):
    logging.info(f"Aggregating responses: {subquery_responses}")
    aggregation_prompt = "Here are the responses to subqueries:\n"
    for i, response in enumerate(subquery_responses, 1):
        aggregation_prompt += f"Subquery {i}: {response}\n"

    aggregation_prompt += "Please combine these responses into one coherent and concise response."

    messages = [
        {"role": "system", "content": aggregation_system_content},
        {"role": "user", "content": aggregation_prompt}
    ]

    try:
        response = await openai_post_request(
            messages, aggregation_model_name, aggregation_max_tokens, aggregation_temperature
        )
        aggregated_response = response['choices'][0]['message']['content'].strip()
        logging.info(f"Final aggregated response: {aggregated_response}")
        return aggregated_response
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP error during aggregation: {str(e)} - {e.response.text}")
        return f"Error: {str(e)}"
    except Exception as e:
        logging.error(f"Error during aggregation: {str(e)}")
        logging.error(traceback.format_exc())
        return f"Error: {str(e)}"

@app.route('/api/message', methods=['POST'])
async def handle_message():
    logging.info("Received request")
    data = request.get_json()
    if not data:
        logging.error("No input data provided")
        return jsonify({'error': 'No input data provided'}), 400

    user_message = data.get('message')
    if not user_message:
        logging.error("No message provided")
        return jsonify({'error': 'No message provided'}), 400

    logging.info(f"Handling message: {user_message}")
    is_complex, subqueries = await analyze_and_decompose_query_with_llm(user_message)

    if is_complex:
        logging.info(f"Subqueries found: {subqueries}")
        tasks = [process_subquery(subquery) for subquery in subqueries]
        assistant_responses = await asyncio.gather(*tasks)

        final_response = await aggregate_responses_with_llm(assistant_responses)
        if "Error" in final_response:
            logging.error(f"Error in final response: {final_response}")
            return jsonify({'error': final_response}), 500
    else:
        if "Error" in subqueries[0]:
            logging.error(f"Error in subqueries: {subqueries[0]}")
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
            logging.info(f"Final response for simple query: {final_response}")
        except httpx.HTTPStatusError as e:
            logging.error(f"HTTP error in simple query: {str(e)}")
            return jsonify({'error': str(e)}), 500
        except Exception as e:
            logging.error(f"General error in simple query: {str(e)}")
            return jsonify({'error': str(e)}), 500

    return jsonify({'response': final_response})

if __name__ == '__main__':
    app.run(debug=False)  # Отключаем режим отладки
