import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server URL configuration
SERVER_URL = os.getenv('SERVER_URL', 'http://localhost:5000')
MESSAGE_ENDPOINT = f'{SERVER_URL}/api/message'
STATUS_ENDPOINT = f'{SERVER_URL}/status'

def check_server_status(timeout=30, interval=1):
    """Checks if the server is ready within the specified timeout period."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(STATUS_ENDPOINT)
            if response.status_code == 200 and response.json().get('status') == 'ready':
                print("Server is ready.")
                return True
        except requests.ConnectionError:
            pass
        print("Waiting for the server to be ready...")
        time.sleep(interval)
    print("Server did not respond within the given time.")
    return False

def send_message(message):
    """Sends a message to the server and prints the assistant's response."""
    try:
        response = requests.post(MESSAGE_ENDPOINT, json={'message': message}, timeout=60)
        response.raise_for_status()
        data = response.json()
        print(f"Assistant: {data['response']}")
    except requests.RequestException as e:
        print(f"Error communicating with the server: {e}")

def main():
    if not check_server_status():
        print("Could not establish a connection with the server.")
        return

    while True:
        user_message = input("You (type 'quit' to exit): ")
        if user_message.lower() == 'quit':
            break
        send_message(user_message)

if __name__ == '__main__':
    main()