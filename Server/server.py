import socket
import threading
from datetime import datetime

class SimpleHTTPServer:
    def __init__(self, host='localhost', port=8080):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
    def start(self):
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        print(f"Server running on http://{self.host}:{self.port}")
        
        while True:
            client_socket, addr = self.socket.accept()
            thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            thread.start()
    
    def handle_client(self, client_socket):
        request = client_socket.recv(1024).decode('utf-8')
        print(f"Request received:\n{request}")
        
        # Парсим HTTP запрос
        lines = request.split('\r\n')
        if not lines:
            return
            
        method, path, _ = lines[0].split(' ')
        
        # Генерируем ответ
        response = self.generate_response(method, path)
        client_socket.send(response.encode('utf-8'))
        client_socket.close()
    
    def generate_response(self, method, path):
        if path == '/':
            content = self.generate_html()
            content_length = len(content)
            
            response = f"""HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Content-Length: {content_length}
Connection: close

{content}"""
        elif path == '/data':
            # Пример JSON API
            content = '{"data": [1, 2, 3, 4, 5], "timestamp": "' + str(datetime.now()) + '"}'
            content_length = len(content)
            
            response = f"""HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: {content_length}
Connection: close

{content}"""
        else:
            # 404 Not Found
            content = "<h1>404 Not Found</h1>"
            content_length = len(content)
            
            response = f"""HTTP/1.1 404 Not Found
Content-Type: text/html; charset=utf-8
Content-Length: {content_length}
Connection: close

{content}"""
        
        return response.replace('\n', '\r\n')
    
    def generate_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Data Reports Server</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Отчеты по обработке данных</h1>
    <div id="chart"></div>
    <div id="data"></div>
    
    <script>
        // Загрузка данных с сервера
        fetch('/data')
            .then(response => response.json())
            .then(data => {
                document.getElementById('data').innerHTML = 
                    'Данные: ' + JSON.stringify(data);
                
                // Создание графика
                Plotly.newPlot('chart', [{
                    y: data.data,
                    type: 'line'
                }]);
            });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    server = SimpleHTTPServer('0.0.0.0', 8080)
    server.start()