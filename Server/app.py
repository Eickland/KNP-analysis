from flask import Flask, render_template, send_from_directory, send_file
import os
from pathlib import Path
import datetime

app = Flask(__name__)

# Папки для данных
DATA_FOLDER = Path('data')
REPORTS_FOLDER = Path('reports')

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
    """Страница аналитики с графиками корреляций"""
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
    # Получаем список HTML отчетов из папки Reports
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
    
    # Сортируем по дате изменения (новые сверху)
    html_reports.sort(key=lambda x: x['modified'], reverse=True)
    
    return render_template('reports.html', 
                         html_reports=html_reports,
                         reports_count=len(html_reports))

@app.route('/reports/files/<filename>')
def serve_report_file(filename):
    """Сервис для отдачи HTML отчетов"""
    return send_from_directory('Reports', filename)

@app.route('/reports/download/<filename>')
def download_report(filename):
    """Скачивание отчета"""
    return send_file(
        REPORTS_FOLDER / filename,
        as_attachment=True,
        download_name=filename
    )

@app.route('/predictions')
def predictions():
    """Страница прогнозов"""
    return render_template('predictions.html')

@app.route('/data-import')
def data_import():
    """Страница импорта данных"""
    return render_template('data_import.html')

@app.route('/settings')
def settings():
    """Страница настроек"""
    return render_template('settings.html')

@app.route('/data/<filename>')
def serve_data_file(filename):
    """Сервис для отдачи файлов из папки Data"""
    return send_from_directory('Data', filename)



if __name__ == '__main__':
    print("Запуск Flask приложения...")
    print("Сайт доступен по адресу: http://localhost:5000")
    
    # Автоматически открываем браузер
    import webbrowser
    webbrowser.open('http://localhost:5000')
    
    # Запускаем Flask приложение
    app.run(
        host='localhost',
        port=5000,
        debug=True
    )