import requests
import time

# Серверный URL
SERVER_URL = 'http://localhost:5000'
MESSAGE_ENDPOINT = f'{SERVER_URL}/api/message'
STATUS_ENDPOINT = f'{SERVER_URL}/status'

def check_server_status(timeout=30, interval=1):
    """Проверяет состояние сервера в течение заданного таймаута."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(STATUS_ENDPOINT)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'ready':
                    print("Сервер готов к работе.")
                    return True
        except requests.ConnectionError:
            pass
        print("Ожидание готовности сервера...")
        time.sleep(interval)
    print("Сервер не ответил в течение заданного времени.")
    return False

def send_message(message):
    payload = {'message': message}
    response = requests.post(MESSAGE_ENDPOINT, json=payload)

    if response.status_code == 200:
        data = response.json()
        print(f"Assistant: {data['response']}")
    else:
        print(f"Request failed with status code {response.status_code}")

if __name__ == '__main__':
    # Проверяем состояние сервера перед отправкой сообщения
    if check_server_status():
        user_message = input("You: ")
        send_message(user_message)
    else:
        print("Не удалось установить соединение с сервером.")