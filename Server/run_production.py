#!/usr/bin/env python3
"""
Скрипт для запуска продакшн сервера
"""

import os
import sys
import logging
from pathlib import Path

# Добавляем текущую директорию в PATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Настройка логирования"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('server.log'),
            logging.StreamHandler()
        ]
    )

def check_dependencies():
    """Проверка зависимостей"""
    try:
        import flask
        import matplotlib
        return True
    except ImportError as e:
        print(f"❌ Отсутствуют зависимости: {e}")
        print("Установите: pip install -r requirements.txt")
        return False

def main():
    """Основная функция запуска"""
    print("🚀 Запуск DataLab Server в продакшн режиме...")
    
    # Настройка логирования
    setup_logging()
    
    # Проверка зависимостей
    if not check_dependencies():
        sys.exit(1)
    
    # Создаем необходимые папки
    Path('data').mkdir(exist_ok=True)
    Path('reports').mkdir(exist_ok=True)
    
    # Запускаем приложение
    from app import app, Config
    
    print(f"📍 Сервер запущен на порту {Config.PORT}")
    print("⚠️  Не закрывайте это окно!")
    
    try:
        app.run(
            host=Config.HOST,
            port=Config.PORT,
            debug=Config.DEBUG
        )
    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен")
    except Exception as e:
        print(f"❌ Ошибка запуска: {e}")
        logging.error(f"Server error: {e}")

if __name__ == '__main__':
    main()