import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from datetime import datetime, timedelta
import sqlite3
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    from sklearn.preprocessing import StandardScaler
except ImportError:
    import sys
    print(f"Python path: {sys.executable}")
    print(f"Installed packages path: {sys.path}")
    raise
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля для matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class MedicalAnalysisSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Система анализа заболеваний населения Казахстана v2.0")
        self.root.geometry("1600x900")
        
        # Инициализация переменных
        self.current_data = None
        self.processed_data = []
        self.forecast_results = None
        self.analysis_cache = {}
        self.ml_models = {}
        
        # Создание и инициализация базы данных
        self.init_database()
        
        # Создание интерфейса
        self.create_menu()
        self.create_toolbar()
        self.create_main_interface()
        self.create_status_bar()
        
        # Применение темы
        self.apply_theme()
        
        # Инициализация моделей ML
        self.init_ml_models()

        # Загрузка данных из базы при старте
        self.load_data_from_db()
        
    def init_database(self):
        """Создание базы данных и заполнение тестовыми данными"""
        self.db_path = "medical_data.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Основные таблицы проекта
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patients_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                region TEXT,
                disease TEXT,
                count INTEGER,
                age INTEGER,
                gender TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS regions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                latitude REAL,
                longitude REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS weather_factors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT,
                month INTEGER,
                avg_temp REAL,
                precipitation REAL
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                forecast_date TEXT,
                region TEXT,
                disease TEXT,
                predicted_count INTEGER,
                actual_count INTEGER
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS report_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT,
                report_type TEXT,
                content TEXT
            )
            """
        )

        conn.commit()

        # Заполнение тестовыми данными при первом запуске
        cursor.execute("SELECT COUNT(*) FROM regions")
        if cursor.fetchone()[0] == 0:
            self._populate_regions(cursor)

        cursor.execute("SELECT COUNT(*) FROM patients_data")
        if cursor.fetchone()[0] == 0:
            df = generate_test_data()
            df.to_sql("patients_data", conn, if_exists="append", index=False)

        cursor.execute("SELECT COUNT(*) FROM weather_factors")
        if cursor.fetchone()[0] == 0:
            weather = generate_weather_data()
            weather.to_sql("weather_factors", conn, if_exists="append", index=False)

        conn.commit()
        conn.close()

    def _populate_regions(self, cursor):
        """Заполнение таблицы regions базовыми данными"""
        regions = [
            ("Алматы", 43.2220, 76.8512),
            ("Астана", 51.1605, 71.4704),
            ("Караганда", 49.8047, 73.1094),
            ("Шымкент", 42.3417, 69.5901),
            ("Актобе", 50.2839, 57.1670),
            ("Павлодар", 52.2873, 76.9674),
            ("Тараз", 42.9000, 71.3667),
            ("Усть-Каменогорск", 49.9480, 82.6176),
            ("Костанай", 53.2144, 63.6246),
            ("Атырау", 47.1076, 51.9142),
        ]
        cursor.executemany(
            "INSERT INTO regions(name, latitude, longitude) VALUES (?, ?, ?)",
            regions,
        )
        
    def init_ml_models(self):
        """Инициализация различных моделей машинного обучения"""
        self.ml_models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
            'Advanced RF': RandomForestRegressor(
                n_estimators=200, 
                max_depth=10, 
                min_samples_split=5,
                random_state=42
            )
        }
        
    def create_menu(self):
        """Создание расширенного главного меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить данные", command=self.load_data)
        file_menu.add_command(label="Импорт из БД", command=self.import_from_database)
        file_menu.add_command(label="Экспорт в БД", command=self.export_to_database)
        file_menu.add_separator()
        file_menu.add_command(label="Сохранить результаты", command=self.save_results)
        file_menu.add_command(label="Сохранить проект", command=self.save_project)
        file_menu.add_command(label="Загрузить проект", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Настройки", command=self.show_settings)
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню Анализ
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Анализ", menu=analysis_menu)
        analysis_menu.add_command(label="Анализ сезонности", command=self.analyze_seasonality)
        analysis_menu.add_command(label="Анализ по регионам", command=self.analyze_regions)
        analysis_menu.add_command(label="Анализ по возрасту", command=self.analyze_age_groups)
        analysis_menu.add_command(label="Анализ корреляций", command=self.analyze_correlation)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Детекция аномалий", command=self.detect_anomalies)
        analysis_menu.add_command(label="Кластерный анализ", command=self.cluster_analysis)
        analysis_menu.add_command(label="Анализ трендов", command=self.trend_analysis)
        
        # Меню Прогноз
        forecast_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Прогноз", menu=forecast_menu)
        forecast_menu.add_command(label="Прогноз SARIMA", command=self.forecast_sarima)
        forecast_menu.add_command(label="Прогноз ML", command=self.forecast_ml)
        forecast_menu.add_command(label="Ансамбль моделей", command=self.ensemble_forecast)
        forecast_menu.add_command(label="Сравнение моделей", command=self.compare_models)
        
        # Меню Качество данных
        quality_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Качество", menu=quality_menu)
        quality_menu.add_command(label="Проверка качества", command=self.check_data_quality)
        quality_menu.add_command(label="Очистка данных", command=self.clean_data)
        quality_menu.add_command(label="Отчет по качеству", command=self.quality_report)
        
        # Меню Справка
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="Руководство пользователя", command=self.show_user_guide)
        help_menu.add_command(label="Горячие клавиши", command=self.show_hotkeys)
        help_menu.add_command(label="О программе", command=self.show_about)

    # Stub methods for menu items
    def import_from_database(self):
        """Импорт данных из базы данных"""
        messagebox.showinfo("Импорт", "Функция импорта из БД в разработке")
        
    def export_to_database(self):
        """Экспорт данных в базу данных"""
        messagebox.showinfo("Экспорт", "Функция экспорта в БД в разработке")
        
    def save_results(self):
        """Сохранение результатов анализа"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel файлы", "*.xlsx"), ("CSV файлы", "*.csv")]
        )
        
        if filename:
            try:
                if filename.endswith('.xlsx'):
                    self.current_data.to_excel(filename, index=False)
                else:
                    self.current_data.to_csv(filename, index=False, encoding='utf-8-sig')
                messagebox.showinfo("Успех", f"Результаты сохранены в {filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")
                
    def save_project(self):
        """Сохранение проекта"""
        messagebox.showinfo("Проект", "Функция сохранения проекта в разработке")
        
    def load_project(self):
        """Загрузка проекта"""
        messagebox.showinfo("Проект", "Функция загрузки проекта в разработке")
        
    def analyze_age_groups(self):
        """Анализ по возрастным группам"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Анализ", "Анализ по возрастным группам в разработке")
        
    def analyze_correlation(self):
        """Анализ корреляций"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Анализ", "Анализ корреляций в разработке")
        
    def cluster_analysis(self):
        """Кластерный анализ"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Анализ", "Кластерный анализ в разработке")
        
    def trend_analysis(self):
        """Анализ трендов"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Анализ", "Анализ трендов в разработке")
        
    def forecast_sarima(self):
        """Прогноз SARIMA"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Прогноз", "Прогноз SARIMA в разработке")
        
    def forecast_ml(self):
        """Прогноз ML"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Прогноз", "Прогноз ML в разработке")
        
    def compare_models(self):
        """Сравнение моделей"""
        messagebox.showinfo("Модели", "Сравнение моделей в разработке")
        
    def clean_data(self):
        """Очистка данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Очистка", "Очистка данных в разработке")
        
    def quality_report(self):
        """Отчет по качеству"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Отчет", "Отчет по качеству в разработке")
        
    def show_user_guide(self):
        """Показ руководства пользователя"""
        guide_window = tk.Toplevel(self.root)
        guide_window.title("Руководство пользователя")
        guide_window.geometry("800x600")
        
        guide_text = tk.Text(guide_window, wrap=tk.WORD, font=("Arial", 11))
        guide_scroll = ttk.Scrollbar(guide_window, command=guide_text.yview)
        guide_text.configure(yscrollcommand=guide_scroll.set)
        
        guide_content = """
РУКОВОДСТВО ПОЛЬЗОВАТЕЛЯ
Система анализа заболеваний населения Казахстана v2.0

1. НАЧАЛО РАБОТЫ
================
- Загрузите данные через меню "Файл" → "Загрузить данные"
- Поддерживаемые форматы: CSV, Excel (.xlsx, .xls), JSON
- Обязательные столбцы: Дата, Регион, Заболевание, Количество

2. АНАЛИЗ ДАННЫХ
================
- Используйте вкладку "Анализ" для различных типов анализа
- Доступны: сезонность, региональный анализ, возрастные группы, корреляции
- Все графики можно сохранять и экспортировать

3. ПРОГНОЗИРОВАНИЕ
==================
- Вкладка "Прогнозы" содержит различные модели прогнозирования
- SARIMA - для временных рядов
- ML модели - для сложных зависимостей
- Ансамблевое прогнозирование - комбинация моделей
        """
        
        guide_text.insert(1.0, guide_content)
        guide_text.config(state=tk.DISABLED)
        
        guide_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        guide_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def show_hotkeys(self):
        """Показ горячих клавиш"""
        messagebox.showinfo("Горячие клавиши", "Горячие клавиши:\nCtrl+O - Загрузить данные\nCtrl+S - Сохранить результаты")
        
    def show_about(self):
        """Информация о программе"""
        about_text = """
Система анализа заболеваний населения Казахстана
Версия 2.0

Основные возможности:
• Загрузка и обработка медицинских данных
• Анализ сезонности и региональных особенностей
• Прогнозирование с использованием ML
• Генерация детальных отчетов

© 2025 Система здравоохранения РК
        """
        messagebox.showinfo("О программе", about_text)
        
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Основные кнопки
        ttk.Button(toolbar, text="📂 Загрузить", command=self.load_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Сохранить", command=self.save_results).pack(side=tk.LEFT, padx=2)
        
        # Разделитель
        ttk.Separator(toolbar, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Инструменты анализа
        ttk.Button(toolbar, text="📊 Анализ", command=self.quick_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📈 Прогноз", command=self.quick_forecast).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔍 Аномалии", command=self.detect_anomalies).pack(side=tk.LEFT, padx=2)
        
        # Индикатор прогресса
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(toolbar, variable=self.progress_var, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        
    def quick_analysis(self):
        """Быстрый анализ"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        self.analyze_seasonality()
        
    def quick_forecast(self):
        """Быстрый прогноз"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
        messagebox.showinfo("Прогноз", "Быстрый прогноз в разработке")
        
    def create_main_interface(self):
        """Создание основного интерфейса"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создание вкладок
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка "Данные"
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="📊 Данные")
        self.create_data_tab()
        
        # Вкладка "Анализ"
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📈 Анализ")
        self.create_analysis_tab()
        
        # Вкладка "Прогнозы"
        self.forecast_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.forecast_frame, text="🔮 Прогнозы")
        self.create_forecast_tab()
        
    def create_data_tab(self):
        """Создание вкладки данных"""
        # Таблица для отображения данных
        columns = ('ID', 'Дата', 'Регион', 'Заболевание', 'Количество')
        self.data_tree = ttk.Treeview(self.data_frame, columns=columns, show='headings', height=15)
        
        # Определение заголовков
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)
        
        # Скроллбары
        vsb = ttk.Scrollbar(self.data_frame, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(self.data_frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Размещение
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        self.data_frame.grid_rowconfigure(0, weight=1)
        self.data_frame.grid_columnconfigure(0, weight=1)
        
        # Панель статистики
        stats_frame = ttk.LabelFrame(self.data_frame, text="Статистика")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=6, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_analysis_tab(self):
        """Создание вкладки анализа"""
        # Панель управления анализом
        control_frame = ttk.LabelFrame(self.analysis_frame, text="Параметры анализа")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Кнопки анализа
        ttk.Button(control_frame, text="Анализ сезонности", command=self.analyze_seasonality).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(control_frame, text="Анализ по регионам", command=self.analyze_regions).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Область для графиков
        self.analysis_plot_frame = ttk.Frame(self.analysis_frame)
        self.analysis_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_forecast_tab(self):
        """Создание вкладки прогнозов"""
        # Панель управления прогнозированием
        control_frame = ttk.LabelFrame(self.forecast_frame, text="Параметры прогнозирования")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Период прогноза
        ttk.Label(control_frame, text="Период прогноза (месяцы):").pack(side=tk.LEFT, padx=5)
        self.forecast_period = tk.IntVar(value=6)
        ttk.Spinbox(control_frame, from_=1, to=24, textvariable=self.forecast_period, width=10).pack(side=tk.LEFT, padx=5)
        
        # Кнопки прогнозирования
        ttk.Button(control_frame, text="Создать прогноз", command=self.ensemble_forecast).pack(side=tk.LEFT, padx=10)
        
        # Область для графиков прогноза
        self.forecast_plot_frame = ttk.Frame(self.forecast_frame)
        self.forecast_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def create_status_bar(self):
        """Создание статусной строки"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(self.status_bar, text="Готов к работе", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.records_label = ttk.Label(self.status_bar, text="Записей: 0", relief=tk.SUNKEN, width=15)
        self.records_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(self.status_bar, text="", relief=tk.SUNKEN, width=20)
        self.time_label.pack(side=tk.LEFT)
        
        self.update_time()
        
    def update_time(self):
        """Обновление времени в статусной строке"""
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
        
    def apply_theme(self):
        """Применение темы оформления"""
        style = ttk.Style()
        style.theme_use('clam')
        
    def load_data(self):
        """Загрузка данных"""
        filename = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[
                ("Все поддерживаемые", "*.csv;*.xlsx;*.xls"),
                ("CSV файлы", "*.csv"), 
                ("Excel файлы", "*.xlsx;*.xls"), 
                ("JSON файлы", "*.json"),
                ("Все файлы", "*.*")
            ]
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    self.current_data = pd.read_csv(filename, encoding='utf-8')
                elif filename.endswith(('.xlsx', '.xls')):
                    self.current_data = pd.read_excel(filename)
                elif filename.endswith('.json'):
                    self.current_data = pd.read_json(filename)
                else:
                    raise ValueError("Неподдерживаемый формат файла")
                
                self.update_data_display()
                self.update_status(f"Загружено {len(self.current_data)} записей из {os.path.basename(filename)}")
                messagebox.showinfo("Успех", "Данные успешно загружены!")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при загрузке файла: {str(e)}")

    def load_data_from_db(self):
        """Загрузка данных из таблицы patients_data"""
        conn = sqlite3.connect(self.db_path)
        try:
            self.current_data = pd.read_sql_query(
                "SELECT date AS 'Дата', region AS 'Регион', disease AS 'Заболевание',"
                " count AS 'Количество', age AS 'Возраст', gender AS 'Пол' FROM patients_data",
                conn,
            )
        finally:
            conn.close()

        if not self.current_data.empty:
            self.update_data_display()
            self.update_status(f"Загружено {len(self.current_data)} записей из базы данных")
                
    def update_data_display(self):
        """Обновление отображения данных"""
        # Очистка таблицы
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        if self.current_data is not None:
            # Заполнение таблицы (показываем первые 100 строк)
            for idx, row in self.current_data.head(100).iterrows():
                values = [idx] + list(row[:4])  # ID + первые 4 колонки
                self.data_tree.insert('', 'end', values=values)
            
            # Обновление статистики
            self.update_statistics()
            
    def update_statistics(self):
        """Обновление статистики"""
        if self.current_data is not None:
            stats_text = f"""
Всего записей: {len(self.current_data):,}
Период: с {pd.to_datetime(self.current_data['Дата']).min().strftime('%Y-%m-%d')} по {pd.to_datetime(self.current_data['Дата']).max().strftime('%Y-%m-%d')}
Уникальных регионов: {self.current_data['Регион'].nunique()}
Типов заболеваний: {self.current_data['Заболевание'].nunique()}
Общее количество случаев: {self.current_data['Количество'].sum():,}
Среднее количество в записи: {self.current_data['Количество'].mean():.1f}
            """
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, stats_text.strip())
            
            self.records_label.config(text=f"Записей: {len(self.current_data):,}")
            
    def analyze_seasonality(self):
        """Анализ сезонности"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Очистка области графиков
        for widget in self.analysis_plot_frame.winfo_children():
            widget.destroy()
            
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Подготовка данных
        data = self.current_data.copy()
        data['Дата'] = pd.to_datetime(data['Дата'])
        data['Месяц'] = data['Дата'].dt.month
        data['Год'] = data['Дата'].dt.year
        
        # График 1: Сезонность по месяцам
        monthly_data = data.groupby('Месяц')['Количество'].sum()
        ax1.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Месяц')
        ax1.set_ylabel('Количество случаев')
        ax1.set_title('Сезонность заболеваемости по месяцам')
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(range(1, 13))
        
        # График 2: Динамика по годам
        yearly_data = data.groupby(['Год', 'Месяц'])['Количество'].sum().unstack(fill_value=0)
        for year in yearly_data.index:
            ax2.plot(range(1, 13), yearly_data.loc[year], marker='o', label=str(year), alpha=0.7)
        ax2.set_xlabel('Месяц')
        ax2.set_ylabel('Количество случаев')
        ax2.set_title('Сравнение сезонности по годам')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Встраивание графика в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_status("Анализ сезонности выполнен")
        
    def analyze_regions(self):
        """Анализ по регионам"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Очистка области графиков
        for widget in self.analysis_plot_frame.winfo_children():
            widget.destroy()
            
        # Создание фигуры
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Данные по регионам
        regional_data = self.current_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
        
        # График 1: Столбчатая диаграмма
        ax1.bar(range(len(regional_data)), regional_data.values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_xticks(range(len(regional_data)))
        ax1.set_xticklabels(regional_data.index, rotation=45, ha='right')
        ax1.set_xlabel('Регион')
        ax1.set_ylabel('Количество случаев')
        ax1.set_title('Заболеваемость по регионам')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # График 2: Круговая диаграмма (топ-5)
        top_regions = regional_data.head(5)
        others = regional_data[5:].sum()
        if others > 0:
            plot_data = pd.concat([top_regions, pd.Series([others], index=['Другие'])])
        else:
            plot_data = top_regions
            
        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        ax2.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Топ-5 регионов по заболеваемости')
        
        plt.tight_layout()
        
        # Встраивание графика
        canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_status("Анализ по регионам выполнен")
        
    def detect_anomalies(self):
        """Детекция аномалий в данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Простой метод детекции аномалий - межквартильный размах
        data = self.current_data['Количество']
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Поиск аномалий
        anomalies = self.current_data[(data < lower_bound) | (data > upper_bound)]
        
        if len(anomalies) > 0:
            messagebox.showinfo("Аномалии", f"Обнаружено {len(anomalies)} аномальных записей")
            
            # Создание окна с аномалиями
            anomaly_window = tk.Toplevel(self.root)
            anomaly_window.title("Обнаруженные аномалии")
            anomaly_window.geometry("800x400")
            
            # Таблица аномалий
            columns = list(self.current_data.columns)
            anomaly_tree = ttk.Treeview(anomaly_window, columns=columns, show='headings')
            
            for col in columns:
                anomaly_tree.heading(col, text=col)
                anomaly_tree.column(col, width=100)
            
            for idx, row in anomalies.iterrows():
                anomaly_tree.insert('', 'end', values=list(row))
            
            anomaly_tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        else:
            messagebox.showinfo("Аномалии", "Аномальных записей не обнаружено")
            
        self.update_status(f"Проверка аномалий завершена. Найдено: {len(anomalies)}")
        
    def ensemble_forecast(self):
        """Простой ансамблевый прогноз"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Очистка области графиков
        for widget in self.forecast_plot_frame.winfo_children():
            widget.destroy()
            
        try:
            # Подготовка данных для прогноза
            data = self.current_data.copy()
            data['Дата'] = pd.to_datetime(data['Дата'])
            
            # Группировка по месяцам
            monthly_data = data.groupby(pd.Grouper(key='Дата', freq='M'))['Количество'].sum()
            monthly_data = monthly_data.dropna()
            
            if len(monthly_data) < 12:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для прогноза (нужно минимум 12 месяцев)")
                return
            
            # Простой прогноз на основе среднего и тренда
            periods = self.forecast_period.get()
            
            # Вычисление тренда (простая линейная регрессия)
            x = np.arange(len(monthly_data))
            y = monthly_data.values
            
            # Коэффициенты линейной регрессии
            slope = np.polyfit(x, y, 1)[0]
            intercept = np.polyfit(x, y, 1)[1]
            
            # Сезонность (среднее по месяцам)
            seasonal_pattern = []
            for month in range(1, 13):
                month_data = data[data['Дата'].dt.month == month]['Количество']
                if len(month_data) > 0:
                    seasonal_pattern.append(month_data.mean())
                else:
                    seasonal_pattern.append(y.mean())
            
            # Создание прогноза
            forecast_values = []
            last_x = len(monthly_data) - 1
            
            for i in range(periods):
                # Линейный тренд
                trend_value = slope * (last_x + i + 1) + intercept
                
                # Сезонная компонента
                month_idx = (monthly_data.index[-1].month + i) % 12
                seasonal_factor = seasonal_pattern[month_idx] / np.mean(seasonal_pattern)
                
                # Прогнозное значение
                forecast_value = max(0, trend_value * seasonal_factor)
                forecast_values.append(forecast_value)
            
            # Создание дат для прогноза
            last_date = monthly_data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                         periods=periods, freq='M')
            
            # Создание графика
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Исторические данные
            ax.plot(monthly_data.index, monthly_data.values, 
                   label='Исторические данные', marker='o', linewidth=2, color='blue')
            
            # Прогноз
            ax.plot(forecast_dates, forecast_values, 
                   label=f'Прогноз на {periods} месяцев', 
                   marker='s', linestyle='--', linewidth=2, color='red')
            
            # Доверительный интервал (простой)
            error_margin = np.std(monthly_data.values) * 0.5
            upper_bound = np.array(forecast_values) + error_margin
            lower_bound = np.array(forecast_values) - error_margin
            
            ax.fill_between(forecast_dates, lower_bound, upper_bound, 
                           alpha=0.3, color='red', label='Доверительный интервал')
            
            ax.set_xlabel('Дата')
            ax.set_ylabel('Количество случаев')
            ax.set_title(f'Прогноз заболеваемости на {periods} месяцев')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Улучшение отображения дат
            fig.autofmt_xdate()
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохранение результатов прогноза
            self.forecast_results = {
                'dates': forecast_dates,
                'values': forecast_values,
                'type': 'ensemble'
            }
            
            self.update_status(f"Прогноз на {periods} месяцев создан успешно")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании прогноза: {str(e)}")
            
    def check_data_quality(self):
        """Проверка качества данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для проверки!")
            return
            
        quality_issues = []
        
        # Проверка пропущенных значений
        missing_values = self.current_data.isnull().sum()
        if missing_values.sum() > 0:
            quality_issues.append(f"Пропущенные значения: {missing_values.sum()}")
        
        # Проверка дубликатов
        duplicates = self.current_data.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"Дубликаты: {duplicates}")
        
        # Проверка отрицательных значений в количестве
        if 'Количество' in self.current_data.columns:
            negative_values = (self.current_data['Количество'] < 0).sum()
            if negative_values > 0:
                quality_issues.append(f"Отрицательные значения: {negative_values}")
        
        # Отображение результатов
        if quality_issues:
            issues_text = "\n".join([f"• {issue}" for issue in quality_issues])
            messagebox.showwarning("Проблемы с качеством данных", 
                                 f"Обнаружены следующие проблемы:\n\n{issues_text}")
        else:
            messagebox.showinfo("Качество данных", "Критических проблем с качеством данных не обнаружено!")
            
        self.update_status("Проверка качества данных завершена")
        
    def show_settings(self):
        """Показ настроек"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Настройки")
        settings_window.geometry("400x300")
        settings_window.transient(self.root)
        
        # Настройки темы
        ttk.Label(settings_window, text="Тема оформления:").pack(pady=10)
        theme_var = tk.StringVar(value="Светлая")
        ttk.Combobox(settings_window, textvariable=theme_var, 
                    values=['Светлая', 'Темная'], state='readonly').pack(pady=5)
        
        # Настройки языка
        ttk.Label(settings_window, text="Язык интерфейса:").pack(pady=10)
        lang_var = tk.StringVar(value="Русский")
        ttk.Combobox(settings_window, textvariable=lang_var, 
                    values=['Русский', 'Қазақша', 'English'], state='readonly').pack(pady=5)
        
        # Кнопки
        button_frame = ttk.Frame(settings_window)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Применить", 
                  command=lambda: self.apply_settings(theme_var.get(), lang_var.get(), settings_window)).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Отмена", 
                  command=settings_window.destroy).pack(side=tk.LEFT, padx=5)
        
    def apply_settings(self, theme, language, window):
        """Применение настроек"""
        messagebox.showinfo("Настройки", f"Настройки применены:\nТема: {theme}\nЯзык: {language}")
        window.destroy()
        
    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=message)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def generate_report(self):
        """Генерация отчета"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для создания отчета!")
            return
        messagebox.showinfo("Отчет", "Функция генерации отчета в разработке")
        
    def export_filtered_data(self):
        """Экспорт отфильтрованных данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта!")
            return
        messagebox.showinfo("Экспорт", "Функция экспорта отфильтрованных данных в разработке")
        
    def quick_report(self):
        """Быстрый отчет"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для отчета!")
            return
        messagebox.showinfo("Отчет", "Функция быстрого отчета в разработке")
        
    def refresh_data(self):
        """Обновление данных"""
        if self.current_data is not None:
            self.update_data_display()
            self.update_status("Данные обновлены")
        
    def apply_filters(self):
        """Применение фильтров"""
        if self.current_data is not None:
            self.update_data_display()
            self.update_status("Фильтры применены")
        
    def reset_filters(self):
        """Сброс фильтров"""
        self.update_status("Фильтры сброшены")
        
    def save_filters(self):
        """Сохранение фильтров"""
        messagebox.showinfo("Фильтры", "Функция сохранения фильтров в разработке")
        
    def export_data(self):
        """Экспорт данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта!")
            return
        self.save_results()


def generate_test_data():
    """Генерация тестовых данных для демонстрации"""
    np.random.seed(42)
    
    # Параметры генерации
    start_date = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    regions = ['Алматы', 'Астана', 'Шымкент', 'Караганда', 'Актобе', 'Павлодар']
    diseases = ['ОРВИ', 'Грипп', 'Пневмония', 'Бронхит', 'Астма', 'Диабет', 'Гипертония']
    
    data = []
    record_id = 1
    
    for date in start_date:
        # Количество записей в день (больше зимой для респираторных заболеваний)
        month = date.month
        winter_factor = 2.0 if month in [12, 1, 2] else 1.0
        daily_records = max(5, int(np.random.poisson(15) * winter_factor))
        
        for _ in range(daily_records):
            region = np.random.choice(regions)
            disease = np.random.choice(diseases)
            
            # Базовое количество случаев
            base_count = np.random.poisson(3)
            
            # Сезонные факторы
            if disease in ['ОРВИ', 'Грипп', 'Пневмония', 'Бронхит'] and month in [12, 1, 2, 3]:
                seasonal_factor = np.random.uniform(2.0, 4.0)
            elif disease in ['Астма'] and month in [4, 5, 9, 10]:
                seasonal_factor = np.random.uniform(1.5, 2.5)
            else:
                seasonal_factor = np.random.uniform(0.8, 1.2)
            
            # Региональные факторы
            regional_factor = {'Алматы': 1.5, 'Астана': 1.3, 'Шымкент': 1.2}.get(region, 1.0)
            
            # Финальное количество
            final_count = max(1, int(base_count * seasonal_factor * regional_factor))
            
            data.append({
                'ID': record_id,
                'Дата': date.strftime('%Y-%m-%d'),
                'Регион': region,
                'Заболевание': disease,
                'Количество': final_count,
                'Возраст': max(0, min(100, int(np.random.normal(45, 20)))),
                'Пол': np.random.choice(['М', 'Ж'])
            })
            
            record_id += 1
    
    return pd.DataFrame(data)


def generate_weather_data():
    """Создает тестовые погодные данные по регионам"""
    np.random.seed(1)
    regions = [
        "Алматы",
        "Астана",
        "Шымкент",
        "Караганда",
        "Актобе",
        "Павлодар",
    ]

    months = range(1, 13)
    data = []
    for region in regions:
        base_temps = np.array([-6, -5, 0, 10, 18, 23, 26, 25, 18, 10, 2, -4])
        base_temps += np.random.normal(0, 1, 12)
        precipitation = np.random.randint(10, 80, 12)
        for m in months:
            data.append({
                "region": region,
                "month": m,
                "avg_temp": float(base_temps[m - 1]),
                "precipitation": int(precipitation[m - 1]),
            })
    return pd.DataFrame(data)


def main():
    """Главная функция запуска приложения"""
    root = tk.Tk()
    
    # Установка иконки (если есть)
    try:
        root.iconbitmap('medical_icon.ico')
    except:
        pass
    
    # Создание приложения
    app = MedicalAnalysisSystem(root)
    
    # Запуск главного цикла
    root.mainloop()


if __name__ == "__main__":
    main()
