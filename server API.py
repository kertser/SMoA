from flask import Flask, request, jsonify
from openai import OpenAI
import configparser
from utility import handle_openai_error, handle_request_error, handle_internal_error
from openai import OpenAIError

config = configparser.ConfigParser()
config.read('config.ini')

# Set OpenAI API key
api_key = config['openai']['api_key']

# Get model configuration
model_name = config['model']['name']
max_tokens = config['model'].getint('max_tokens')
temperature = config['model'].getfloat('temperature')

# Get system prompt
system_content = config['prompt']['system_content']

# Initialize the Flask app
app = Flask(__name__)
client = OpenAI(
    api_key=api_key,
)

# Register error handlers
app.register_error_handler(OpenAIError, handle_openai_error)
app.register_error_handler(400, handle_request_error)
app.register_error_handler(500, handle_internal_error)

@app.route('/api/message', methods=['POST'])
def handle_message():
    data = request.get_json()
    if not data:
        return handle_request_error(ValueError("No input data provided"))

    user_message = data.get('message')
    if not user_message:
        return handle_request_error(ValueError("No message provided"))

    # Prepare the messages for ChatCompletion
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message}
    ]

    # Call the OpenAI GPT model
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_reply = response.choices[0].message.content.strip()
    except OpenAIError as e:
        return handle_openai_error(e)
    except Exception as e:
        return handle_internal_error(e)

    # Return the response to the client
    return jsonify({'response': assistant_reply})

if __name__ == '__main__':
    app.run(debug=True)