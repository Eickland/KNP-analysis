from flask import Flask, render_template, send_from_directory, send_file, request, abort, Response
from pathlib import Path
import datetime
import secrets
import time
import webbrowser
import threading
import socket

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Для безопасности сессий

# Конфигурация
class Config:
    HOST = '0.0.0.0'  # Слушаем все интерфейсы
    PORT = 5000
    DEBUG = False
    # Ограничения безопасности
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'html'}

app.config.from_object(Config)

# Папки для данных
DATA_FOLDER = Path('Received data','Compare data','Graphs')
REPORTS_FOLDER = Path('reports')
REPORTS_FOLDER.mkdir(exist_ok=True)


def open_browser():
    """Функция для открытия браузера"""
    time.sleep(1.5)  # Ждем немного, чтобы сервер успел запуститься
    webbrowser.open_new('http://127.0.0.1:5000/')

# Маршруты
@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Страница дашборда"""
    return render_template('dashboard.html')

@app.route('/analytics')
def analytics():
    """Страница аналитики"""
    correlation_images = []
    if DATA_FOLDER.exists():
        for file in DATA_FOLDER.glob('*.png'):
            correlation_images.append(file.name)
    
    return render_template('analytics.html', 
                         correlation_images=correlation_images,
                         images_count=len(correlation_images))

@app.route('/reports')
def reports():
    """Страница отчетов"""
    html_reports = []
    if REPORTS_FOLDER.exists():
        for file in REPORTS_FOLDER.glob('*.html'):
            report_info = {
                'filename': file.name,
                'size': file.stat().st_size,
                'modified': datetime.datetime.fromtimestamp(file.stat().st_mtime),
                'url': f"/reports/files/{file.name}"
            }
            html_reports.append(report_info)
    
    html_reports.sort(key=lambda x: x['modified'], reverse=True)
    
    return render_template('reports.html', 
                         html_reports=html_reports,
                         reports_count=len(html_reports))

# Остальные маршруты с @requires_auth...
@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/data-import')
def data_import():
    return render_template('data_import.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# Статические файлы (без авторизации для CSS/JS)
@app.route('/Received data/Compare data/Graphs/<filename>')
def serve_data_file(filename):
    return send_from_directory('Received data/Compare data/Graphs', filename)

@app.route('/reports/files/<filename>')
def serve_report_file(filename):
    return send_from_directory('Reports', filename)

@app.route('/reports/download/<filename>')
def download_report(filename):
    return send_file(REPORTS_FOLDER / filename, as_attachment=True)

# Health check (без авторизации)
@app.route('/health')
def health():
    return {'status': 'ok', 'timestamp': datetime.datetime.now().isoformat()}

if __name__ == '__main__':
    print("=" * 50)
    print("DataLab Server - Test Mode")
    print("=" * 50)
    
    
    # Информация о доступе

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"📡 Сервер будет доступен по:")
    print(f"   Локально: http://localhost:{Config.PORT}")
    print("=" * 50)
    
    # Запускаем тестовый сервер
    threading.Thread(target=open_browser).start()
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )
