import os
from pathlib import Path

class ProductionConfig:
    # Безопасность
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    
    # Сетевые настройки
    HOST = '0.0.0.0'
    PORT = 5000
    
    # Ограничения
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = Path('uploads')
    
    # Аутентификация
    USERS = {
        'admin': 'our792lpQ!',  # ИЗМЕНИТЕ!
        'analyst': 'Kx361fgr_2'  # ИЗМЕНИТЕ!
    }
    
    # IP фильтрация (оставьте пустым для разрешения всех IP)
    ALLOWED_IPS = {
        # '192.168.1.100',  # Пример: разрешить только конкретный IP
        # Добавьте ваш белый IP и IP доверенных пользователей
    }
    
    # Логирование
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'server.log'

class DevelopmentConfig(ProductionConfig):
    DEBUG = True
    LOG_LEVEL = 'DEBUG'