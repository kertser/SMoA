import requests

# The server's URL
SERVER_URL = 'http://localhost:5000/api/message'


def send_message(message):
    # Prepare the payload
    payload = {'message': message}

    # Send the POST request to the server API
    response = requests.post(SERVER_URL, json=payload)

    if response.status_code == 200:
        # Extract the assistant's response
        data = response.json()
        assistant_reply = data.get('response', '')
        print(f"Assistant: {assistant_reply}")
    else:
        print(f"Request failed with status code {response.status_code}")


if __name__ == '__main__':
    user_message = input("You: ")
    send_message(user_message)