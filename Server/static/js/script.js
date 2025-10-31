// Основной модуль приложения
class DataLabApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupScrollEffects();
        this.setupButtonAnimations();
        console.log('DataLab App initialized');
    }

    setupEventListeners() {
        // Обработчик для кнопок навигации
        document.querySelectorAll('.section-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleSectionClick(e));
        });

        // Обработчик для мобильного меню
        const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
        const navLinks = document.querySelector('.nav-links');
        
        if (mobileMenuBtn) {
            mobileMenuBtn.addEventListener('click', () => {
                navLinks.classList.toggle('active');
            });
        }

        // Закрытие мобильного меню при клике на ссылку
        document.querySelectorAll('.nav-links a').forEach(link => {
            link.addEventListener('click', () => {
                navLinks.classList.remove('active');
            });
        });
    }

    setupScrollEffects() {
        let lastScrollTop = 0;
        const header = document.querySelector('header');

        window.addEventListener('scroll', () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            
            // Эффект уменьшения хедера при скролле
            if (scrollTop > 100) {
                header.classList.add('header-scrolled');
            } else {
                header.classList.remove('header-scrolled');
            }

            // Параллакс эффект для героя
            const hero = document.querySelector('.hero');
            if (hero) {
                const scrolled = window.pageYOffset;
                const rate = scrolled * -0.5;
                hero.style.transform = `translateY(${rate}px)`;
            }

            lastScrollTop = scrollTop;
        });
    }

    setupButtonAnimations() {
        // Анимация при наведении на кнопки
        document.querySelectorAll('.section-btn').forEach(btn => {
            btn.addEventListener('mouseenter', (e) => {
                this.animateButton(e.currentTarget, 'enter');
            });
            
            btn.addEventListener('mouseleave', (e) => {
                this.animateButton(e.currentTarget, 'leave');
            });
        });
    }

    animateButton(button, action) {
        if (action === 'enter') {
            button.style.transform = 'translateY(-10px) scale(1.05)';
        } else {
            button.style.transform = 'translateY(0) scale(1)';
        }
    }

    handleSectionClick(event) {
        event.preventDefault();
        const button = event.currentTarget;
        const sectionName = button.querySelector('.btn-title').textContent;
        
        // Показываем уведомление
        this.showNotification(`Переход в раздел: ${sectionName}`);
        
        // Имитация загрузки
        this.simulateLoading(button).then(() => {
            // Здесь будет реальный переход на другую страницу
            console.log(`Navigating to: ${sectionName}`);
            // window.location.href = button.getAttribute('href');
        });
    }

    showNotification(message) {
        // Создаем или находим элемент уведомления
        let notification = document.querySelector('.notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.className = 'notification';
            document.body.appendChild(notification);
        }
        
        notification.textContent = message;
        notification.classList.add('show');
        
        // Автоматическое скрытие через 3 секунды
        setTimeout(() => {
            notification.classList.remove('show');
        }, 3000);
    }

    simulateLoading(button) {
        return new Promise((resolve) => {
            const originalContent = button.innerHTML;
            button.innerHTML = '<div class="loading"></div>';
            button.style.pointerEvents = 'none';
            
            setTimeout(() => {
                button.innerHTML = originalContent;
                button.style.pointerEvents = 'auto';
                resolve();
            }, 1500);
        });
    }

    // Метод для динамической загрузки данных
    async loadData(endpoint) {
        try {
            const response = await fetch(endpoint);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error loading data:', error);
            this.showNotification('Ошибка загрузки данных');
        }
    }
}

// Инициализация приложения после загрузки DOM
document.addEventListener('DOMContentLoaded', () => {
    new DataLabApp();
});

// Дополнительные утилиты
const utils = {
    // Форматирование чисел
    formatNumber: (num) => {
        return new Intl.NumberFormat('ru-RU').format(num);
    },

    // Форматирование даты
    formatDate: (date) => {
        return new Intl.DateTimeFormat('ru-RU').format(new Date(date));
    },

    // Генератор случайного цвета
    getRandomColor: () => {
        const colors = ['#00d4ff', '#0099cc', '#ff6b6b', '#4ecdc4', '#45b7d1'];
        return colors[Math.floor(Math.random() * colors.length)];
    }
};

// Экспорт для использования в других модулях
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { DataLabApp, utils };
}
