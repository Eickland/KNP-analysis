from flask import Flask, render_template, send_from_directory, send_file, request, abort, Response
import os
from pathlib import Path
import datetime
from functools import wraps
import secrets

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
DATA_FOLDER = Path('Data')
REPORTS_FOLDER = Path('Reports')
REPORTS_FOLDER.mkdir(exist_ok=True)

# База пользователей (в продакшн заменить на реальную БД)
USERS = {
    'admin': 'our792lpQ!',  # Замените на сложный пароль
    'user': 'Kx361fgr_2'
}

def check_auth(username, password):
    """Проверка логина и пароля"""
    return USERS.get(username) == password

def requires_auth(f):
    """Декоратор для требующих аутентификации страниц"""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def authenticate():
    """Отправляет запрос аутентификации"""
    return Response(
        'Требуется авторизация\n'
        'Введите логин и пароль',
        401,
        {'WWW-Authenticate': 'Basic realm="DataLab Analytics"'}
    )

# IP фильтрация (опционально)
ALLOWED_IPS = {}

def check_ip():
    """Проверка IP адреса (опционально)"""
    if not ALLOWED_IPS:  # Если список пустой - разрешаем все
        return True
    
    client_ip = request.remote_addr
    return client_ip in ALLOWED_IPS

@app.before_request
def limit_remote_addr():
    """Ограничение доступа по IP"""
    if not check_ip():
        abort(403)  # Forbidden

# Маршруты
@app.route('/')
@requires_auth  # Требует авторизации
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/dashboard')
@requires_auth
def dashboard():
    """Страница дашборда"""
    return render_template('dashboard.html')

@app.route('/analytics')
@requires_auth
def analytics():
    """Страница аналитики"""
    correlation_images = []
    if DATA_FOLDER.exists():
        for file in DATA_FOLDER.glob('*.png'):
            correlation_images.append(file.name)
    
    if not correlation_images:
        correlation_images = ["correlation_matrix.png", "feature_correlations.png"]
    
    return render_template('analytics.html', 
                         correlation_images=correlation_images,
                         images_count=len(correlation_images))

@app.route('/reports')
@requires_auth
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
@requires_auth
def predictions():
    return render_template('predictions.html')

@app.route('/data-import')
@requires_auth
def data_import():
    return render_template('data_import.html')

@app.route('/settings')
@requires_auth
def settings():
    return render_template('settings.html')

# Статические файлы (без авторизации для CSS/JS)
@app.route('/data/<filename>')
@requires_auth
def serve_data_file(filename):
    return send_from_directory('Data', filename)

@app.route('/reports/files/<filename>')
@requires_auth
def serve_report_file(filename):
    return send_from_directory('Reports', filename)

@app.route('/reports/download/<filename>')
@requires_auth
def download_report(filename):
    return send_file(REPORTS_FOLDER / filename, as_attachment=True)

# Health check (без авторизации)
@app.route('/health')
def health():
    return {'status': 'ok', 'timestamp': datetime.datetime.now().isoformat()}

if __name__ == '__main__':
    print("=" * 50)
    print("DataLab Server - Production Mode")
    print("=" * 50)
    
    
    # Информация о доступе
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"📡 Сервер будет доступен по:")
    print(f"   Локально: http://localhost:{Config.PORT}")
    print(f"   В сети: http://{local_ip}:{Config.PORT}")
    print(f"   Глобально: http://ВАШ_БЕЛЫЙ_IP:{Config.PORT}")
    print(f"🔐 Требуется авторизация")
    print(f"   Логин: admin, Пароль: password123")
    print("=" * 50)
    
    # Запускаем продакшн сервер
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG,
        threaded=True
    )
