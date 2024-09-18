import requests
import time

# Server URL configuration
SERVER_URL = 'http://localhost:5000'
MESSAGE_ENDPOINT = f'{SERVER_URL}/api/message'
STATUS_ENDPOINT = f'{SERVER_URL}/status'

def check_server_status(timeout=30, interval=1):
    """Checks if the server is ready within the specified timeout period.

    Args:
        timeout (int): Maximum time to wait for the server to be ready, in seconds.
        interval (int): Time interval between status checks, in seconds.

    Returns:
        bool: True if the server is ready, False if the timeout is reached.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Send a GET request to the server's status endpoint
            response = requests.get(STATUS_ENDPOINT)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ready':
                    print("Server is ready.")
                    return True
        except requests.ConnectionError:
            # Ignore connection errors and retry
            pass
        print("Waiting for the server to be ready...")
        time.sleep(interval)
    print("Server did not respond within the given time.")
    return False

def send_message(message):
    """Sends a message to the server and prints the assistant's response.

    Args:
        message (str): The user's message to send to the assistant.
    """
    payload = {'message': message}
    # Send a POST request to the server with the user's message
    response = requests.post(MESSAGE_ENDPOINT, json=payload)

    if response.status_code == 200:
        data = response.json()
        # Output the assistant's response
        print(f"Assistant: {data['response']}")
    else:
        print(f"Request failed with status code {response.status_code}")

if __name__ == '__main__':
    # Check the server status before sending a message
    if check_server_status():
        # Prompt the user for input
        user_message = input("You: ")
        # Send the message to the assistant
        send_message(user_message)
    else:
        print("Could not establish a connection with the server.")
