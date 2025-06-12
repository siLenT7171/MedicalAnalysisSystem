#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
import webbrowser
import tempfile
from fpdf import FPDF
import google.generativeai as genai
import warnings
warnings.filterwarnings('ignore')

# Проверка scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False

# Проверка openpyxl
try:
    import openpyxl
    from openpyxl.styles import Font, Alignment
    OPENPYXL_AVAILABLE = True
except ImportError as e:
    OPENPYXL_AVAILABLE = False

# Проверка XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError as e:
    XGBOOST_AVAILABLE = False

# Проверка statsmodels
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    STATSMODELS_AVAILABLE = False

# Настройка стиля для matplotlib
plt.style.use('default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class MedicalAnalysisSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Система анализа заболеваний населения Казахстана v1.2 (Исправленная)")
        self.root.geometry("1400x800")
        self.gemini_api_key = None
        # Инициализация переменных
        self.current_data = None
        self.processed_data = None
        self.forecast_results = None
        
        # Создание базы данных
        self.init_database()
        
        # Создание интерфейса
        self.create_menu()
        self.create_toolbar()
        self.create_main_interface()
        self.create_status_bar()
        
        # Применение темы
        self.apply_theme()
        
    def init_database(self):
        """Инициализация базы данных SQLite"""
        try:
            self.db_path = "medical_data.db"
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Создание таблиц
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    analysis_type TEXT,
                    parameters TEXT,
                    results TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            messagebox.showerror("Ошибка БД", f"Ошибка инициализации базы данных: {str(e)}")
        
    def create_menu(self):
        """Создание главного меню"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Меню Файл
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить данные", command=self.load_data)
        file_menu.add_command(label="Сохранить результаты", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.root.quit)
        
        # Меню Анализ
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Анализ", menu=analysis_menu)
        analysis_menu.add_command(label="Анализ сезонности", command=self.analyze_seasonality)
        analysis_menu.add_command(label="Анализ по регионам", command=self.analyze_regions)
        analysis_menu.add_command(label="Анализ по возрасту", command=self.analyze_age_groups)
        analysis_menu.add_command(label="Анализ корреляций", command=self.analyze_correlation)
        
        # Меню Прогноз
        forecast_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Прогноз", menu=forecast_menu)
        forecast_menu.add_command(label="Прогноз SARIMA", command=self.forecast_sarima)
        if SKLEARN_AVAILABLE:
            forecast_menu.add_command(label="Прогноз ML (Random Forest)", command=self.forecast_ml)
            forecast_menu.add_command(label="Линейная регрессия", command=self.forecast_linear_regression)
        if XGBOOST_AVAILABLE:
            forecast_menu.add_command(label="Прогноз XGBoost", command=self.forecast_xgboost)
        
    def create_toolbar(self):
        """Создание панели инструментов"""
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Кнопки инструментов
        ttk.Button(toolbar, text="📂 Загрузить", command=self.load_data).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="💾 Сохранить", command=self.save_results).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📊 Анализ", command=self.quick_analysis).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🤖 Gemini Анализ", command=self.analyze_with_gemini).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📈 Прогноз", command=self.quick_forecast).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="📄 Отчет", command=self.generate_report).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="🔄 Сброс фильтров", command=self.reset_filters).pack(side=tk.LEFT, padx=2)
        
    def create_main_interface(self):
        """Создание основного интерфейса с вкладками"""
        # Главный контейнер
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создание вкладок
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка "Данные"
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Данные")
        self.create_data_tab()
        
        # Вкладка "Анализ"
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="Анализ")
        self.create_analysis_tab()
        
        # Вкладка "Прогнозы"
        self.forecast_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.forecast_frame, text="Прогнозы")
        self.create_forecast_tab()
        
        # Вкладка "Отчеты"
        self.report_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.report_frame, text="Отчеты")
        self.create_report_tab()
        
        # Вкладка "Карты" 
        self.map_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.map_frame, text="Карты")
        self.create_map_tab()
        
    def apply_theme(self):
        """Применение темы оформления"""
        style = ttk.Style()
        
        # Выбираем доступную тему
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        else:
            style.theme_use(available_themes[0])
        
        # Настройка цветов
        try:
            style.configure('TLabel', background='#f0f0f0')
            style.configure('TFrame', background='#f0f0f0')
            style.configure('TLabelframe', background='#f0f0f0')
            style.configure('TLabelframe.Label', background='#f0f0f0', foreground='#333333')
        except:
            pass  # Игнорируем ошибки настройки стилей
        
    def create_data_tab(self):
        """Создание вкладки данных"""
        # Панель управления
        control_panel = ttk.LabelFrame(self.data_frame, text="Фильтры данных")
        control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Первая строка фильтров
        filter_frame1 = ttk.Frame(control_panel)
        filter_frame1.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(filter_frame1, text="Период с:").pack(side=tk.LEFT, padx=5)
        self.date_from = ttk.Entry(filter_frame1, width=12)
        self.date_from.pack(side=tk.LEFT, padx=2)
        self.date_from.insert(0, "2020-01-01")  # Значение по умолчанию
        
        ttk.Label(filter_frame1, text="по:").pack(side=tk.LEFT, padx=5)
        self.date_to = ttk.Entry(filter_frame1, width=12)
        self.date_to.pack(side=tk.LEFT, padx=2)
        self.date_to.insert(0, "2024-12-31")  # Значение по умолчанию
        
        ttk.Label(filter_frame1, text="Регион:").pack(side=tk.LEFT, padx=(15, 5))
        self.region_var = tk.StringVar(value="Все")
        self.region_combo = ttk.Combobox(filter_frame1, textvariable=self.region_var, width=15)
        self.region_combo['values'] = ['Все']
        self.region_combo.pack(side=tk.LEFT, padx=2)
        
        # Вторая строка с кнопками
        filter_frame2 = ttk.Frame(control_panel)
        filter_frame2.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(filter_frame2, text="Применить фильтры", command=self.apply_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame2, text="Сбросить фильтры", command=self.reset_filters).pack(side=tk.LEFT, padx=5)
        ttk.Button(filter_frame2, text="Экспорт данных", command=self.export_filtered_data).pack(side=tk.LEFT, padx=5)
        
        # Информационная панель
        info_frame = ttk.Frame(filter_frame2)
        info_frame.pack(side=tk.RIGHT, padx=5)
        ttk.Label(info_frame, text="💡 Формат даты: YYYY-MM-DD", foreground="gray").pack()
        
        # Таблица данных
        self.create_data_table()
        
        # Панель статистики
        stats_panel = ttk.LabelFrame(self.data_frame, text="Статистика")
        stats_panel.pack(fill=tk.X, padx=5, pady=5)
        
        self.stats_text = tk.Text(stats_panel, height=6, wrap=tk.WORD, state=tk.DISABLED)
        stats_scroll = ttk.Scrollbar(stats_panel, orient="vertical", command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
    def create_data_table(self):
        """Создание таблицы для отображения данных"""
        # Фрейм для таблицы
        table_frame = ttk.LabelFrame(self.data_frame, text="Таблица данных")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создание Treeview
        columns = ('ID', 'Дата', 'Регион', 'Заболевание', 'Возраст', 'Пол', 'Количество')
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Определение заголовков и ширины колонок
        column_widths = {'ID': 50, 'Дата': 100, 'Регион': 120, 'Заболевание': 120, 
                        'Возраст': 80, 'Пол': 50, 'Количество': 100}
        
        for col in columns:
            self.data_tree.heading(col, text=col, command=lambda c=col: self.sort_treeview(c))
            self.data_tree.column(col, width=column_widths.get(col, 100), minwidth=50)
        
        # Скроллбары
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Размещение
        self.data_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        
        # Информация о таблице
        info_label = ttk.Label(table_frame, text="Показано первые 1000 записей. Используйте фильтры для уточнения.")
        info_label.grid(row=2, column=0, columnspan=2, pady=5)

    def create_status_bar(self):
        """Создание статусной строки"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Статус
        self.status_label = ttk.Label(self.status_bar, text="Готов к работе", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Счетчик записей
        self.records_label = ttk.Label(self.status_bar, text="Записей: 0", relief=tk.SUNKEN, width=15)
        self.records_label.pack(side=tk.LEFT, padx=2)
        
        # Время
        self.time_label = ttk.Label(self.status_bar, text="", relief=tk.SUNKEN, width=20)
        self.time_label.pack(side=tk.LEFT, padx=2)
        
        # Версия
        version_label = ttk.Label(self.status_bar, text="v1.2", relief=tk.SUNKEN, width=8)
        version_label.pack(side=tk.LEFT, padx=2)
        
        self.update_time()
        
    def update_time(self):
        """Обновление времени в статусной строке"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def sort_treeview(self, col):
        """Сортировка данных в таблице по колонке"""
        try:
            # Получаем все элементы
            data = [(self.data_tree.set(child, col), child) for child in self.data_tree.get_children('')]
            
            # Определяем тип сортировки
            if col in ['ID', 'Возраст', 'Количество']:
                # Числовая сортировка
                data.sort(key=lambda x: float(x[0]) if x[0].replace('.', '').replace('-', '').isdigit() else 0)
            else:
                # Текстовая сортировка
                data.sort(key=lambda x: x[0].lower())
            
            # Переупорядочиваем элементы
            for index, (val, child) in enumerate(data):
                self.data_tree.move(child, '', index)
                
            self.update_status(f"Таблица отсортирована по: {col}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сортировке: {str(e)}")
    
    def reset_filters(self):
        """Сброс всех фильтров"""
        try:
            self.date_from.delete(0, tk.END)
            self.date_from.insert(0, "2020-01-01")

            self.date_to.delete(0, tk.END)
            self.date_to.insert(0, "2024-12-31")

            self.region_var.set("Все")

            if hasattr(self, 'analysis_region_var'):
                self.analysis_region_var.set('Все')
            if hasattr(self, 'analysis_date_from'):
                self.analysis_date_from.delete(0, tk.END)
                self.analysis_date_from.insert(0, "2020-01-01")
            if hasattr(self, 'analysis_date_to'):
                self.analysis_date_to.delete(0, tk.END)
                self.analysis_date_to.insert(0, "2024-12-31")
            if hasattr(self, 'forecast_region_var'):
                self.forecast_region_var.set('Все')
            
            # Показываем все данные
            if self.current_data is not None:
                self.update_data_display()
                self.update_status("Фильтры сброшены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сбросе фильтров: {str(e)}")
    
    def export_filtered_data(self):
        """Экспорт отфильтрованных данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта!")
            return
        
        data_to_export = self.processed_data if self.processed_data is not None else self.current_data
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Excel файлы", "*.xlsx")],
            title="Экспорт отфильтрованных данных"
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    data_to_export.to_csv(filename, index=False, encoding='utf-8')
                else:
                    data_to_export.to_excel(filename, index=False)
                    
                messagebox.showinfo("Успех", f"Данные экспортированы: {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при экспорте: {str(e)}")
                
    def create_map_tab(self):
        """Создание вкладки с картами (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        # Панель управления картой
        map_control_panel = ttk.LabelFrame(self.map_frame, text="Параметры карты")
        map_control_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Выбор типа карты
        ttk.Label(map_control_panel, text="Тип карты:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.map_type = tk.StringVar(value="regional")
        map_types = [
            ("По регионам", "regional"),
            ("Плотность населения", "density"),
            ("Временная динамика", "temporal")
        ]
        
        col = 1
        for text, value in map_types:
            ttk.Radiobutton(map_control_panel, text=text, variable=self.map_type, 
                        value=value).grid(row=0, column=col, padx=5)
            col += 1
        
        # Выбор показателя (ИСПРАВЛЕНО)
        ttk.Label(map_control_panel, text="Показатель:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.map_metric = tk.StringVar()
        self.metric_combo = ttk.Combobox(map_control_panel, textvariable=self.map_metric, width=20, state="readonly")
        
        # Устанавливаем значения по умолчанию
        default_metrics = ['Всего случаев', 'На 100К населения', 'Темп роста', 'Средняя тяжесть']
        self.metric_combo['values'] = default_metrics
        self.metric_combo.set('Всего случаев')  # Устанавливаем значение по умолчанию
        self.metric_combo.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Временной период (ИСПРАВЛЕНО)
        ttk.Label(map_control_panel, text="Период:").grid(row=1, column=2, padx=5, pady=5, sticky='w')
        self.map_period = tk.StringVar()
        self.period_combo = ttk.Combobox(map_control_panel, textvariable=self.map_period, width=15, state="readonly")
        
        # Устанавливаем значения по умолчанию
        default_periods = ['2024', '2023', '2022', '2021', '2020', 'Все годы']
        self.period_combo['values'] = default_periods
        self.period_combo.set('2024')  # Устанавливаем значение по умолчанию
        self.period_combo.grid(row=1, column=3, padx=5, pady=5, sticky='w')
        
        # Выбор заболевания для анализа
        ttk.Label(map_control_panel, text="Заболевание:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.map_disease = tk.StringVar()
        self.disease_map_combo = ttk.Combobox(map_control_panel, textvariable=self.map_disease, width=20, state="readonly")
        
        # Устанавливаем значения по умолчанию
        self.disease_map_combo['values'] = ['Все']
        self.disease_map_combo.set('Все')
        self.disease_map_combo.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Кнопки управления
        ttk.Button(map_control_panel, text="Построить карту", 
                command=self.build_map).grid(row=2, column=2, padx=10, pady=5)
        
        ttk.Button(map_control_panel, text="Обновить фильтры", 
                command=self.update_map_filters).grid(row=2, column=3, padx=5, pady=5)
        
        # Область для карты
        self.map_plot_frame = ttk.LabelFrame(self.map_frame, text="Интерактивная карта")
        self.map_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель легенды и статистики
        legend_panel = ttk.LabelFrame(self.map_frame, text="Легенда и статистика")
        legend_panel.pack(fill=tk.X, padx=5, pady=5)
        
        self.map_stats_text = tk.Text(legend_panel, height=4, wrap=tk.WORD)
        map_stats_scroll = ttk.Scrollbar(legend_panel, orient="vertical", command=self.map_stats_text.yview)
        self.map_stats_text.configure(yscrollcommand=map_stats_scroll.set)
        
        self.map_stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        map_stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def update_map_filters(self):
        """Обновление фильтров на карте на основе загруженных данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        try:
            # Сохраняем текущие выборы
            current_metric = self.map_metric.get()
            current_period = self.map_period.get()
            current_disease = self.map_disease.get()
            
            # Обновляем список показателей
            metrics = ['Всего случаев', 'На 100К населения', 'Темп роста', 'Средняя тяжесть']
            self.metric_combo['values'] = metrics
            if current_metric in metrics:
                self.map_metric.set(current_metric)
            else:
                self.map_metric.set('Всего случаев')
            
            # Обновляем список заболеваний
            if 'Заболевание' in self.current_data.columns:
                diseases = ['Все'] + sorted(self.current_data['Заболевание'].dropna().unique().tolist())
                self.disease_map_combo['values'] = diseases
                if current_disease in diseases:
                    self.map_disease.set(current_disease)
                else:
                    self.map_disease.set('Все')
            
            # Обновляем список периодов на основе реальных данных
            if 'Дата' in self.current_data.columns:
                try:
                    dates = pd.to_datetime(self.current_data['Дата'], errors='coerce')
                    years = sorted(dates.dt.year.dropna().unique(), reverse=True)
                    year_list = [str(int(year)) for year in years] + ['Все годы']
                    self.period_combo['values'] = year_list
                    
                    if current_period in year_list:
                        self.map_period.set(current_period)
                    elif len(years) > 0:
                        self.map_period.set(str(int(years[0])))
                    else:
                        self.map_period.set('2024')
                except Exception as e:
                    print(f"Ошибка обработки дат: {e}")
                    # Устанавливаем значения по умолчанию
                    default_periods = ['2024', '2023', '2022', '2021', '2020', 'Все годы']
                    self.period_combo['values'] = default_periods
                    if current_period in default_periods:
                        self.map_period.set(current_period)
                    else:
                        self.map_period.set('2024')
            
            self.update_status("Фильтры карты обновлены на основе загруженных данных")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении фильтров карты: {str(e)}")
            # В случае ошибки устанавливаем значения по умолчанию
            self.metric_combo['values'] = ['Всего случаев', 'На 100К населения', 'Темп роста', 'Средняя тяжесть']
            self.metric_combo.set('Всего случаев')
            self.period_combo['values'] = ['2024', '2023', '2022', '2021', '2020', 'Все годы']
            self.period_combo.set('2024')
            self.disease_map_combo['values'] = ['Все']
            self.disease_map_combo.set('Все')
    
    def create_analysis_tab(self):
        """Создание вкладки анализа"""
        # Панель выбора типа анализа
        analysis_panel = ttk.LabelFrame(self.analysis_frame, text="Параметры анализа")
        analysis_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Тип анализа
        ttk.Label(analysis_panel, text="Тип анализа:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.analysis_type = tk.StringVar(value="seasonality")
        analysis_types = [
            ("Сезонность", "seasonality"),
            ("По регионам", "regions"),
            ("По возрастам", "age_groups"),
            ("Корреляция", "correlation")
        ]
        
        col = 1
        for text, value in analysis_types:
            ttk.Radiobutton(analysis_panel, text=text, variable=self.analysis_type, 
                          value=value).grid(row=0, column=col, padx=5, sticky='w')
            col += 1
        
        # Заболевание для анализа
        ttk.Label(analysis_panel, text="Заболевание:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.disease_var = tk.StringVar(value="Все")
        self.disease_combo = ttk.Combobox(analysis_panel, textvariable=self.disease_var, width=20)
        self.disease_combo['values'] = ['Все']
        self.disease_combo.grid(row=1, column=1, columnspan=2, padx=5, pady=5, sticky='w')
        
        ttk.Button(analysis_panel, text="Выполнить анализ", 
                  command=self.perform_analysis).grid(row=1, column=3, padx=5, pady=5)
        
        ttk.Button(analysis_panel, text="Сохранить график",
                  command=self.save_analysis_plot).grid(row=1, column=4, padx=5, pady=5)

        # Регион для анализа
        ttk.Label(analysis_panel, text="Регион:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.analysis_region_var = tk.StringVar(value="Все")
        self.analysis_region_combo = ttk.Combobox(analysis_panel, textvariable=self.analysis_region_var, width=20)
        self.analysis_region_combo['values'] = ['Все']
        self.analysis_region_combo.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky='w')

        # Период анализа
        ttk.Label(analysis_panel, text="Период с:").grid(row=2, column=3, padx=5, pady=5, sticky='e')
        self.analysis_date_from = ttk.Entry(analysis_panel, width=12)
        self.analysis_date_from.grid(row=2, column=4, padx=2, pady=5, sticky='w')
        self.analysis_date_from.insert(0, "2020-01-01")

        ttk.Label(analysis_panel, text="по:").grid(row=2, column=5, padx=5, pady=5, sticky='e')
        self.analysis_date_to = ttk.Entry(analysis_panel, width=12)
        self.analysis_date_to.grid(row=2, column=6, padx=2, pady=5, sticky='w')
        self.analysis_date_to.insert(0, "2024-12-31")

        # Область для графиков
        self.analysis_plot_frame = ttk.LabelFrame(self.analysis_frame, text="Результаты анализа")
        self.analysis_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Добавляем начальную информацию
        info_label = ttk.Label(self.analysis_plot_frame, 
                              text="Выберите тип анализа и нажмите 'Выполнить анализ'", 
                              font=('Arial', 12))
        info_label.pack(expand=True)
        
        try:
            import xgboost as xgb
            XGBOOST_AVAILABLE = True
        except ImportError:
            XGBOOST_AVAILABLE = False
            print("Внимание: XGBoost не установлен. XGBoost прогнозирование будет недоступно.")

        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            STATSMODELS_AVAILABLE = True
        except ImportError:
            STATSMODELS_AVAILABLE = False
            print("Внимание: statsmodels не установлен. SARIMA прогнозирование будет ограничено.")

    def analyze_with_gemini(self):
            if self.current_data is None:
                messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
                return

            # --- ВАШ API КЛЮЧ ---
            # Вставьте свой ключ для Gemini API прямо сюда.
            # ВНИМАНИЕ: Не делитесь этим файлом, если в нем указан ваш ключ.
            api_key = "AIzaSyC45u9u5NOi2VWUbEvyGWp1Ow2cg0MVS6A"

            if api_key == "YOUR_GEMINI_API_KEY_HERE":
                messagebox.showerror("Ключ не найден", "Пожалуйста, вставьте ваш Gemini API ключ в код в методе 'analyze_with_gemini'.")
                return

            # Преобразуем часть данных в текст
            try:
                sample = self.current_data.head(50).to_markdown(index=False)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка преобразования данных: {str(e)}")
                return

            # Подключение Gemini
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')

                # --- ИЗМЕНЕННЫЙ ПРОМПТ ---
                # Добавлена инструкция по форматированию списков
                prompt = f"""
                Проанализируй следующие медицинские данные из Казахстана.
                Твоя задача - выступить в роли эксперта по анализу данных здравоохранения.

                Определи ключевые закономерности, выяви аномалии или выбросы, и предложи 2-3 обоснованные гипотезы для проверки.
                Структурируй свой ответ четко. **Для всех списков используй дефисы (-) вместо звездочек (*).**

                Вот структура ответа:
                1.  **Ключевые закономерности:**
                    - (Твой анализ здесь)
                2.  **Обнаруженные аномалии:**
                    - (Твой анализ здесь)
                3.  **Гипотезы для проверки:**
                    - (Твои гипотезы здесь)

                Вот данные для анализа:
                {sample}
                """

                response = model.generate_content(prompt)
                # Дополнительная обработка для замены оставшихся звездочек на всякий случай
                analysis = response.text.replace('*', '-')

                # Показываем результат
                self.show_text_window("Результат анализа Gemini", analysis)

            except Exception as e:
                messagebox.showerror("Ошибка Gemini", f"Ошибка при вызове Gemini API: {str(e)}\n\n"
                                                    "Убедитесь, что вы ввели правильный API ключ и имеете доступ к интернету.")

    def show_text_window(self, title, content):
        window = tk.Toplevel(self.root)
        window.title(title)
        text_area = tk.Text(window, wrap='word')
        text_area.insert(tk.END, content)
        text_area.config(state=tk.DISABLED)
        text_area.pack(expand=True, fill='both')


    def check_libraries_status(self):
        """Проверка статуса библиотек в runtime"""
        status = {}
        
        # Проверка XGBoost
        try:
            import xgboost as xgb
            status['xgboost'] = True
        except ImportError:
            status['xgboost'] = False
        
        # Проверка scikit-learn
        try:
            from sklearn.ensemble import RandomForestRegressor
            status['sklearn'] = True
        except ImportError:
            status['sklearn'] = False
        
        # Проверка statsmodels
        try:
            from statsmodels.tsa.arima.model import ARIMA
            status['statsmodels'] = True
        except ImportError:
            status['statsmodels'] = False
        
        # Проверка openpyxl
        try:
            import openpyxl
            status['openpyxl'] = True
        except ImportError:
            status['openpyxl'] = False
            
        return status

    def show_library_status(self):
        """Показать статус всех библиотек с macOS советами"""
        status = self.check_libraries_status()
        
        status_text = "📊 СТАТУС БИБЛИОТЕК:\n" + "="*40 + "\n"
        
        libraries = {
            'sklearn': 'scikit-learn (машинное обучение)', 
            'xgboost': 'XGBoost (градиентный бустинг)',
            'statsmodels': 'statsmodels (временные ряды)',
            'openpyxl': 'openpyxl (экспорт в Excel)'
        }
        
        available_count = 0
        for key, name in libraries.items():
            if status[key]:
                icon = "✅"
                available_count += 1
            else:
                icon = "❌"
            status_text += f"{icon} {name}\n"
        
        # Специальное сообщение для macOS и XGBoost
        import platform
        if not status['xgboost'] and platform.system() == "Darwin":
            status_text += "\n🍎 ПРОБЛЕМА macOS + XGBoost:\n"
            status_text += "Установите OpenMP: brew install libomp\n"
        
        status_text += f"\n📈 ПРОГРЕСС: {available_count}/4 библиотек установлено\n\n"
        
        status_text += "🚀 ДОСТУПНЫЕ МОДЕЛИ:\n"
        status_text += "✅ SARIMA (всегда доступна)\n"
        
        if status['sklearn']:
            status_text += "✅ Linear Regression\n"
            status_text += "✅ Random Forest\n"
        else:
            status_text += "❌ Linear Regression (нужен scikit-learn)\n"
            status_text += "❌ Random Forest (нужен scikit-learn)\n"
        
        if status['xgboost']:
            status_text += "✅ XGBoost\n"
        else:
            status_text += "❌ XGBoost (нужен OpenMP: brew install libomp)\n"
        
        status_text += "\n🔧 ДИАГНОСТИКА: Нажмите 'Да' для подробной диагностики"
            
        # Показываем статус и предлагаем диагностику
        result = messagebox.askyesno("Статус библиотек", status_text)
        
        if result:
            self.diagnose_xgboost_issue()
        
    def diagnose_xgboost_issue(self):
        """Диагностика проблем с XGBoost"""
        import platform
        import sys
        
        diag_text = f"""
    ╔══════════════════════════════════════════════════════╗
    ║                  ДИАГНОСТИКА XGBOOST                 ║
    ╚══════════════════════════════════════════════════════╝

    🖥️  Система: {platform.system()} {platform.release()}
    🐍  Python: {sys.version}
    📍  Архитектура: {platform.machine()}

    🔍 СТАТУС БИБЛИОТЕК:
    """
        
        # Проверяем каждую библиотеку отдельно
        libraries = {
            'XGBoost': 'xgboost',
            'scikit-learn': 'sklearn',
            'statsmodels': 'statsmodels',
            'OpenMP (для XGBoost)': None  # Специальная проверка
        }
        
        for name, module in libraries.items():
            if module is None:
                # Проверка OpenMP
                try:
                    import xgboost as xgb
                    # Проверяем, что можем создать модель
                    test_model = xgb.XGBRegressor(n_estimators=1)
                    XGBOOST_AVAILABLE = True
                    print("✅ XGBoost загружен успешно")
                except ImportError as e:
                    XGBOOST_AVAILABLE = False
                    print(f"❌ XGBoost не установлен: {e}")
                except Exception as e:
                    XGBOOST_AVAILABLE = False
                    error_msg = str(e)
                    if "OpenMP" in error_msg or "libomp" in error_msg:
                        print("❌ XGBoost: Ошибка OpenMP runtime на macOS")
                        print("💡 Решение: brew install libomp")
                    else:
                        print(f"❌ XGBoost: Ошибка загрузки - {e}")
        
        if platform.system() == "Darwin":  # macOS
            diag_text += f"""

    🍎 СПЕЦИАЛЬНО ДЛЯ macOS:
    ━━━━━━━━━━━━━━━━━━━━━━━━━

    Если XGBoost не работает, выполните:

    1️⃣  Установите Homebrew (если нет):
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    2️⃣  Установите OpenMP:
    brew install libomp

    3️⃣  Переустановите XGBoost:
    pip uninstall xgboost
    pip install xgboost

    4️⃣  Перезапустите программу

    🔄 Альтернативные варианты:
    • Используйте Random Forest вместо XGBoost
    • Все остальные функции работают без XGBoost
    """
        
        diag_text += f"""

    💡 РЕКОМЕНДАЦИИ:
    • SARIMA: Всегда доступна
    • Linear Regression: {'✅' if globals().get('SKLEARN_AVAILABLE', False) else '❌'}
    • Random Forest: {'✅' if globals().get('SKLEARN_AVAILABLE', False) else '❌'}
    • XGBoost: {'✅' if globals().get('XGBOOST_AVAILABLE', False) else '❌'}

    ⚠️  ВАЖНО: Программа полностью функциональна без XGBoost!
    """
        
        messagebox.showinfo("Диагностика XGBoost", diag_text)
        
    def create_forecast_tab(self):
        """Создание вкладки прогнозов (компактная версия)"""
        # Панель параметров
        forecast_panel = ttk.LabelFrame(self.forecast_frame, text="Параметры прогнозирования")
        forecast_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Все элементы в одной строке
        ttk.Label(forecast_panel, text="Модель:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        self.model_var = tk.StringVar(value="SARIMA")
        model_combo = ttk.Combobox(forecast_panel, textvariable=self.model_var, width=15, state="readonly")
        available_models = ['SARIMA', 'Linear Regression', 'Random Forest', 'XGBoost']
        model_combo['values'] = available_models
        model_combo.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(forecast_panel, text="Период (месяцев):").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        
        self.forecast_period = tk.IntVar(value=6)
        period_spin = ttk.Spinbox(forecast_panel, from_=1, to=24, textvariable=self.forecast_period, width=8)
        period_spin.grid(row=0, column=3, padx=5, pady=5, sticky='w')
        
        # Разделитель
        ttk.Label(forecast_panel, text="Регион:").grid(row=0, column=4, padx=5, pady=5, sticky='w')
        self.forecast_region_var = tk.StringVar(value="Все")
        self.forecast_region_combo = ttk.Combobox(forecast_panel, textvariable=self.forecast_region_var, width=20)
        self.forecast_region_combo['values'] = ['Все']
        self.forecast_region_combo.grid(row=0, column=5, padx=5, pady=5, sticky='w')

        ttk.Button(forecast_panel, text="🚀 Построить прогноз",
                command=self.build_forecast).grid(row=0, column=6, padx=10, pady=5)
        
        # Область для результатов
        self.forecast_plot_frame = ttk.LabelFrame(self.forecast_frame, text="Результаты прогнозирования")
        self.forecast_plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Информационная панель
        info_frame = ttk.Frame(self.forecast_plot_frame)
        info_frame.pack(expand=True, fill='both')
        
        info_label = ttk.Label(info_frame, 
                            text="🎯 СИСТЕМА ПРОГНОЗИРОВАНИЯ ЗАБОЛЕВАЕМОСТИ\n\n"
                                "📈 Выберите модель и настройте параметры прогноза\n"
                                "🔧 Проверьте статус библиотек ML при необходимости\n"
                                "🚀 Нажмите 'Построить прогноз' для запуска анализа\n\n"
                                "📊 Доступные алгоритмы:\n"
                                "• SARIMA: Временные ряды с сезонностью и трендом\n"
                                "• Linear Regression: Линейная регрессия с сезонными факторами\n" 
                                "• Random Forest: Ансамбль деревьев решений\n"
                                "• XGBoost: Градиентный бустинг высокой точности", 
                            font=('Arial', 12), justify='center', foreground='#2c3e50')
        info_label.pack(expand=True)
        
        # Нижняя информационная строка
        bottom_info = ttk.Label(info_frame, 
                            text="💡 Совет: Для данных с явной сезонностью используйте SARIMA, для сложных паттернов - XGBoost", 
                            font=('Arial', 10), foreground='#7f8c8d', justify='center')
        bottom_info.pack(side='bottom', pady=10)
        
    def create_report_tab(self):
        """Создание вкладки отчетов"""
        # Панель настроек отчета
        report_panel = ttk.LabelFrame(self.report_frame, text="Параметры отчета")
        report_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Тип отчета
        ttk.Label(report_panel, text="Тип отчета:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.report_type = tk.StringVar(value="summary")
        report_types = [
            ("Сводный", "summary"),
            ("Детальный", "detailed"),
            ("Прогнозный", "forecast"),
            ("Сравнительный", "comparative")
        ]
        
        col = 1
        for text, value in report_types:
            ttk.Radiobutton(report_panel, text=text, variable=self.report_type, 
                          value=value).grid(row=0, column=col, padx=5, sticky='w')
            col += 1
        
        # Формат экспорта
        ttk.Label(report_panel, text="Формат:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.export_format = tk.StringVar(value="HTML")
        format_combo = ttk.Combobox(report_panel, textvariable=self.export_format, width=10)
        format_combo['values'] = ['HTML', 'PDF', 'Excel', 'Word', 'TXT']
        format_combo.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Button(report_panel, text="Создать отчет", 
                  command=self.create_report).grid(row=1, column=2, padx=10, pady=5)
        ttk.Button(report_panel, text="Экспорт", 
                  command=self.export_report).grid(row=1, column=3, padx=5, pady=5)
        
        # Область предварительного просмотра
        preview_frame = ttk.LabelFrame(self.report_frame, text="Предварительный просмотр")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем текстовое поле с прокруткой
        text_frame = ttk.Frame(preview_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.report_text = tk.Text(text_frame, wrap=tk.WORD, font=('Courier New', 10))
        report_scroll = ttk.Scrollbar(text_frame, command=self.report_text.yview)
        self.report_text.configure(yscrollcommand=report_scroll.set)
        
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        report_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Добавляем начальный текст
        welcome_text = """
        СИСТЕМА АНАЛИЗА ЗАБОЛЕВАНИЙ НАСЕЛЕНИЯ КАЗАХСТАНА
        ================================================
        
        Добро пожаловать в систему анализа медицинских данных!
        
        Для создания отчета:
        1. Загрузите данные (вкладка "Данные")
        2. Выберите тип отчета
        3. Нажмите "Создать отчет"
        4. При необходимости экспортируйте в нужном формате
        
        Доступные типы отчетов:
        • Сводный - общая статистика по данным
        • Детальный - подробный анализ по регионам и периодам
        • Прогнозный - результаты прогнозирования
        • Сравнительный - сравнительный анализ регионов и периодов
        """
        
        self.report_text.insert(1.0, welcome_text)
        self.report_text.config(state=tk.DISABLED)
        
    def load_data(self):
        """Загрузка данных из файла"""
        filename = filedialog.askopenfilename(
            title="Выберите файл данных",
            filetypes=[
                ("CSV файлы", "*.csv"), 
                ("Excel файлы", "*.xlsx *.xls"), 
                ("Все файлы", "*.*")
            ]
        )
        
        if filename:
            try:
                # Показываем прогресс
                self.update_status("Загрузка данных...")
                self.root.update()
                
                # Определяем тип файла и загружаем
                if filename.lower().endswith('.csv'):
                    # Пробуем разные кодировки
                    encodings = ['utf-8', 'cp1251', 'latin-1']
                    data_loaded = False
                    
                    for encoding in encodings:
                        try:
                            self.current_data = pd.read_csv(filename, encoding=encoding)
                            data_loaded = True
                            break
                        except UnicodeDecodeError:
                            continue
                    
                    if not data_loaded:
                        raise ValueError("Не удалось определить кодировку файла")
                        
                elif filename.lower().endswith(('.xlsx', '.xls')):
                    self.current_data = pd.read_excel(filename)
                else:
                    raise ValueError("Неподдерживаемый формат файла")

                # Автопреобразование распространенных альтернативных колонок
                if 'Дата' not in self.current_data.columns and 'Дата диагноза' in self.current_data.columns:
                    self.current_data['Дата'] = self.current_data['Дата диагноза']
                if 'Заболевание' not in self.current_data.columns and 'Код МКБ-10' in self.current_data.columns:
                    self.current_data['Заболевание'] = self.current_data['Код МКБ-10']
                if 'Количество' not in self.current_data.columns:
                    # Если каждая строка представляет один случай
                    self.current_data['Количество'] = 1
                if 'ID' not in self.current_data.columns:
                    self.current_data.reset_index(inplace=True)
                    self.current_data.rename(columns={'index': 'ID'}, inplace=True)
                
                # Валидация данных
                is_valid, message = self.validate_data_format(self.current_data)
                if not is_valid:
                    messagebox.showerror("Ошибка данных", f"Проблема с форматом данных:\n{message}")
                    return
                
                # Обновляем интерфейс
                self.update_data_display()
                self.update_filters()
                self.update_map_filters()  # Обновляем фильтры карты
                
                self.update_status(f"Загружено {len(self.current_data)} записей из {os.path.basename(filename)}")
                messagebox.showinfo("Успех", f"Данные успешно загружены!\nЗаписей: {len(self.current_data)}")
                
            except Exception as e:
                error_msg = f"Ошибка при загрузке файла:\n{str(e)}"
                messagebox.showerror("Ошибка", error_msg)
                self.update_status("Ошибка загрузки данных")
    
    def validate_data_format(self, data):
        """Валидация формата загруженных данных"""
        required_columns = ['Дата', 'Регион', 'Заболевание', 'Количество']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            return False, f"Отсутствуют обязательные колонки: {', '.join(missing_columns)}"
        
        # Проверка типов данных
        try:
            pd.to_datetime(data['Дата'])
        except:
            return False, "Неверный формат колонки 'Дата'. Ожидается формат даты."
        
        if not pd.api.types.is_numeric_dtype(data['Количество']):
            try:
                data['Количество'] = pd.to_numeric(data['Количество'], errors='coerce')
                if data['Количество'].isna().all():
                    return False, "Колонка 'Количество' не содержит числовых значений"
            except:
                return False, "Колонка 'Количество' должна содержать числовые значения"
        
        return True, "Данные корректны"

    def apply_filters(self):
        """Исправленный метод применения фильтров к данным"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для фильтрации!")
            return
            
        try:
            filtered_data = self.current_data.copy()
            
            # Фильтр по региону
            region_filter = self.region_var.get()
            if region_filter and region_filter != 'Все' and region_filter in filtered_data['Регион'].values:
                filtered_data = filtered_data[filtered_data['Регион'] == region_filter]
            
            # Фильтр по дате
            date_from = self.date_from.get().strip()
            date_to = self.date_to.get().strip()
            
            if date_from:
                try:
                    date_from_parsed = pd.to_datetime(date_from)
                    filtered_data = filtered_data[pd.to_datetime(filtered_data['Дата']) >= date_from_parsed]
                except Exception as e:
                    messagebox.showwarning("Предупреждение", 
                                         f"Неверный формат даты 'от': {date_from}. Используйте YYYY-MM-DD")
                    return
            
            if date_to:
                try:
                    date_to_parsed = pd.to_datetime(date_to)
                    filtered_data = filtered_data[pd.to_datetime(filtered_data['Дата']) <= date_to_parsed]
                except Exception as e:
                    messagebox.showwarning("Предупреждение", 
                                         f"Неверный формат даты 'до': {date_to}. Используйте YYYY-MM-DD")
                    return
            
            # Проверяем, есть ли данные после фильтрации
            if len(filtered_data) == 0:
                messagebox.showinfo("Информация", "После применения фильтров данные не найдены")
                return
            
            self.processed_data = filtered_data
            
            # Обновляем отображение отфильтрованных данных
            self.update_filtered_data_display(filtered_data)
            
            self.update_status(f"Применены фильтры. Показано {len(filtered_data)} из {len(self.current_data)} записей")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при применении фильтров: {str(e)}")
    
    def update_filtered_data_display(self, filtered_data):
        """Обновление отображения отфильтрованных данных"""
        try:
            # Очистка таблицы
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            # Заполнение таблицы отфильтрованными данными (первые 1000 записей)
            display_data = filtered_data.head(1000)
            
            for idx, row in display_data.iterrows():
                # Преобразуем все значения в строки для безопасного отображения
                values = []
                for col in ['ID', 'Дата', 'Регион', 'Заболевание', 'Возраст', 'Пол', 'Количество']:
                    if col in row.index:
                        value = row[col]
                        if pd.isna(value):
                            values.append('')
                        else:
                            values.append(str(value))
                    else:
                        values.append('')
                
                self.data_tree.insert('', 'end', values=values)
            
            # Обновление статистики для отфильтрованных данных
            self.update_filtered_statistics(filtered_data)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении отображения: {str(e)}")
                
    def update_data_display(self):
        """Обновление отображения данных в таблице"""
        try:
            # Очистка таблицы
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)
            
            if self.current_data is not None:
                # Заполнение таблицы (показываем первые 1000 записей)
                display_data = self.current_data.head(1000)
                
                for idx, row in display_data.iterrows():
                    values = []
                    for col in ['ID', 'Дата', 'Регион', 'Заболевание', 'Возраст', 'Пол', 'Количество']:
                        if col in row.index:
                            value = row[col]
                            if pd.isna(value):
                                values.append('')
                            else:
                                values.append(str(value))
                        else:
                            values.append('')
                    
                    self.data_tree.insert('', 'end', values=values)
                
                # Обновление статистики
                self.update_statistics()
                
                # Обновление фильтров
                self.update_filters()
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обновлении отображения: {str(e)}")
            
    def update_filters(self):
        """Обновление доступных фильтров"""
        try:
            if self.current_data is not None:
                # Обновление списка регионов
                if 'Регион' in self.current_data.columns:
                    regions = ['Все'] + sorted(self.current_data['Регион'].dropna().unique().tolist())
                    self.region_combo['values'] = regions
                    if self.region_var.get() not in regions:
                        self.region_var.set('Все')

                    if hasattr(self, 'forecast_region_combo'):
                        self.forecast_region_combo['values'] = regions
                        if self.forecast_region_var.get() not in regions:
                            self.forecast_region_var.set('Все')

                    if hasattr(self, 'analysis_region_combo'):
                        self.analysis_region_combo['values'] = regions
                        if self.analysis_region_var.get() not in regions:
                            self.analysis_region_var.set('Все')
                
                # Обновление списка заболеваний
                if 'Заболевание' in self.current_data.columns:
                    diseases = ['Все'] + sorted(self.current_data['Заболевание'].dropna().unique().tolist())
                    self.disease_combo['values'] = diseases
                    if self.disease_var.get() not in diseases:
                        self.disease_var.set('Все')
                        
        except Exception as e:
            print(f"Ошибка при обновлении фильтров: {e}")

    def update_status(self, message):
        """Обновление статусной строки"""
        self.status_label.config(text=message)
        
    def update_statistics(self):
        """Обновление статистической информации"""
        try:
            if self.current_data is not None:
                self.stats_text.config(state=tk.NORMAL)
                self.stats_text.delete(1.0, tk.END)
                
                # Основная статистика
                total_records = len(self.current_data)
                
                # Период данных
                try:
                    dates = pd.to_datetime(self.current_data['Дата'])
                    date_min = dates.min().strftime('%Y-%m-%d')
                    date_max = dates.max().strftime('%Y-%m-%d')
                except:
                    date_min = date_max = 'Неизвестно'
                
                # Уникальные значения
                unique_regions = self.current_data['Регион'].nunique() if 'Регион' in self.current_data.columns else 0
                unique_diseases = self.current_data['Заболевание'].nunique() if 'Заболевание' in self.current_data.columns else 0
                total_cases = self.current_data['Количество'].sum() if 'Количество' in self.current_data.columns else 0
                
                stats_text = f"""ОБЩАЯ СТАТИСТИКА ДАННЫХ
═══════════════════════════════════

📊 Основные показатели:
• Всего записей: {total_records:,}
• Период данных: {date_min} ─ {date_max}
• Уникальных регионов: {unique_regions}
• Типов заболеваний: {unique_diseases}
• Общее количество случаев: {total_cases:,}

📈 Структура данных:
• Колонок: {len(self.current_data.columns)}
• Размер данных: {self.current_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} МБ"""

                # Добавляем топ регионы если есть данные
                if 'Регион' in self.current_data.columns and 'Количество' in self.current_data.columns:
                    try:
                        top_regions = self.current_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False).head(3)
                        stats_text += f"\n\n🏆 ТОП-3 РЕГИОНА:"
                        for i, (region, count) in enumerate(top_regions.items(), 1):
                            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                            stats_text += f"\n{medal} {region}: {count:,}"
                    except:
                        pass
                
                self.stats_text.insert(1.0, stats_text)
                self.stats_text.config(state=tk.DISABLED)
                
                self.records_label.config(text=f"Записей: {len(self.current_data)}")
                
        except Exception as e:
            error_text = f"Ошибка при расчете статистики: {str(e)}"
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, error_text)
            self.stats_text.config(state=tk.DISABLED)

    def update_filtered_statistics(self, filtered_data):
        """Обновление статистики для отфильтрованных данных"""
        try:
            self.stats_text.config(state=tk.NORMAL)
            self.stats_text.delete(1.0, tk.END)
            
            # Проверяем наличие необходимых колонок
            required_cols = ['Дата', 'Регион', 'Заболевание', 'Количество']
            missing_cols = [col for col in required_cols if col not in filtered_data.columns]
            
            if missing_cols:
                stats_text = f"Отсутствуют необходимые колонки: {', '.join(missing_cols)}"
            else:
                total_records = len(filtered_data)
                
                # Даты
                try:
                    dates = pd.to_datetime(filtered_data['Дата'])
                    date_min = dates.min().strftime('%Y-%m-%d') if not dates.isna().all() else 'Не определено'
                    date_max = dates.max().strftime('%Y-%m-%d') if not dates.isna().all() else 'Не определено'
                except:
                    date_min = date_max = 'Ошибка формата'
                
                # Регионы и заболевания
                unique_regions = filtered_data['Регион'].nunique()
                unique_diseases = filtered_data['Заболевание'].nunique()
                
                # Общее количество случаев
                total_cases = filtered_data['Количество'].sum() if 'Количество' in filtered_data.columns else 0
                
                stats_text = f"""СТАТИСТИКА ОТФИЛЬТРОВАННЫХ ДАННЫХ
═══════════════════════════════════════

📊 Основные показатели:
• Записей: {total_records:,}
• Период: {date_min} ─ {date_max}
• Уникальных регионов: {unique_regions}
• Типов заболеваний: {unique_diseases}
• Общее количество случаев: {total_cases:,}

🏆 ТОП-3 РЕГИОНА:"""

                # Топ регионы
                if 'Регион' in filtered_data.columns and 'Количество' in filtered_data.columns:
                    try:
                        top_regions = filtered_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False).head(3)
                        for i, (region, count) in enumerate(top_regions.items(), 1):
                            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                            stats_text += f"\n{medal} {region}: {count:,}"
                    except:
                        stats_text += "\nОшибка расчета"
                
                # Добавляем статистику по заболеваниям
                if 'Заболевание' in filtered_data.columns:
                    try:
                        stats_text += f"\n\n💊 ТОП-3 ЗАБОЛЕВАНИЯ:"
                        top_diseases = filtered_data.groupby('Заболевание')['Количество'].sum().sort_values(ascending=False).head(3)
                        for i, (disease, count) in enumerate(top_diseases.items(), 1):
                            medal = "🔴" if i == 1 else "🟡" if i == 2 else "🟢"
                            stats_text += f"\n{medal} {disease}: {count:,}"
                    except:
                        pass
            
            self.stats_text.insert(1.0, stats_text)
            self.stats_text.config(state=tk.DISABLED)
            
            # Обновляем счетчик записей
            self.records_label.config(text=f"Записей: {len(filtered_data)}")

        except Exception as e:
            error_text = f"Ошибка при расчете статистики: {str(e)}"
            self.stats_text.insert(1.0, error_text)
            self.stats_text.config(state=tk.DISABLED)

    def get_analysis_filtered_data(self):
        """Возвращает данные, отфильтрованные для анализа"""
        if self.current_data is None:
            return pd.DataFrame()

        data = self.current_data.copy()

        # Фильтр по заболеванию
        disease_filter = self.disease_var.get()
        if disease_filter and disease_filter != 'Все' and 'Заболевание' in data.columns:
            data = data[data['Заболевание'] == disease_filter]

        # Фильтр по региону
        if hasattr(self, 'analysis_region_var') and 'Регион' in data.columns:
            region_filter = self.analysis_region_var.get()
            if region_filter and region_filter != 'Все':
                data = data[data['Регион'] == region_filter]

        # Фильтр по дате
        if hasattr(self, 'analysis_date_from') and hasattr(self, 'analysis_date_to'):
            date_from = self.analysis_date_from.get().strip()
            date_to = self.analysis_date_to.get().strip()

            if date_from:
                try:
                    date_from_parsed = pd.to_datetime(date_from)
                    data = data[pd.to_datetime(data['Дата']) >= date_from_parsed]
                except Exception:
                    messagebox.showwarning(
                        "Предупреждение",
                        f"Неверный формат даты 'с': {date_from}. Используйте YYYY-MM-DD")
                    return pd.DataFrame()

            if date_to:
                try:
                    date_to_parsed = pd.to_datetime(date_to)
                    data = data[pd.to_datetime(data['Дата']) <= date_to_parsed]
                except Exception:
                    messagebox.showwarning(
                        "Предупреждение",
                        f"Неверный формат даты 'по': {date_to}. Используйте YYYY-MM-DD")
                    return pd.DataFrame()

        return data

    def build_map(self):
        """Построение карты заболеваемости (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        try:
            # Очистка области карты
            for widget in self.map_plot_frame.winfo_children():
                widget.destroy()
                
            map_type = self.map_type.get()
            
            if map_type == "regional":
                self.build_regional_map()
            elif map_type == "density":
                self.build_density_map()
            elif map_type == "temporal":
                self.build_temporal_map()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении карты: {str(e)}")

    def build_regional_map(self):
        """Построение региональной карты (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        try:
            # Подготовка данных
            period = self.map_period.get()
            metric = self.map_metric.get()
            disease_filter = self.map_disease.get()
            
            data = self.current_data.copy()
            region_filter = getattr(self, 'forecast_region_var', None)
            if region_filter and region_filter != 'Все' and 'Регион' in data.columns:
                data = data[data['Регион'] == region_filter]
            
            # Фильтр по заболеванию
            if disease_filter != 'Все' and 'Заболевание' in data.columns:
                data = data[data['Заболевание'] == disease_filter]
            
            # Фильтр по периоду
            if period != 'Все годы' and period != '':
                try:
                    data['Год'] = pd.to_datetime(data['Дата']).dt.year
                    data = data[data['Год'] == int(period)]
                except:
                    pass
            
            if len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return
            
            # Агрегация по регионам в зависимости от выбранного показателя
            if metric == 'Всего случаев':
                regional_data = data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
                title_suffix = "Общее количество случаев"
                color_label = "Количество случаев"
            elif metric == 'На 100К населения':
                # Примерные данные населения по регионам (в тысячах)
                population_data = {
                    'Алматы': 2000, 'Астана': 1200, 'Караганда': 700, 'Шымкент': 1000,
                    'Актобе': 500, 'Павлодар': 350, 'Тараз': 400, 'Усть-Каменогорск': 350,
                    'Костанай': 250, 'Атырау': 300, 'Петропавловск': 200, 'Актау': 200,
                    'Кокшетау': 150, 'Семей': 350, 'Талдыкорган': 200
                }
                regional_totals = data.groupby('Регион')['Количество'].sum()
                regional_data = pd.Series({region: (count / population_data.get(region, 500)) * 100 
                                         for region, count in regional_totals.items()}).sort_values(ascending=False)
                title_suffix = "На 100К населения"
                color_label = "Случаев на 100К"
            elif metric == 'Темп роста':
                # Расчет темпа роста за последние 2 года
                data['Год'] = pd.to_datetime(data['Дата']).dt.year
                yearly_data = data.groupby(['Регион', 'Год'])['Количество'].sum().unstack(fill_value=0)
                if yearly_data.shape[1] >= 2:
                    last_year = yearly_data.columns[-1]
                    prev_year = yearly_data.columns[-2]
                    regional_data = ((yearly_data[last_year] - yearly_data[prev_year]) / 
                                   yearly_data[prev_year].replace(0, 1) * 100).sort_values(ascending=False)
                    title_suffix = "Темп роста (%)"
                    color_label = "Темп роста (%)"
                else:
                    regional_data = data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
                    title_suffix = "Данных недостаточно для темпа роста"
                    color_label = "Количество случаев"
            else:
                regional_data = data.groupby('Регион')['Количество'].mean().sort_values(ascending=False)
                title_suffix = "Средняя тяжесть"
                color_label = "Среднее значение"
            
            # Создание графика
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            
            # График 1: Столбчатая диаграмма
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(regional_data)))
            bars = ax1.bar(range(len(regional_data)), regional_data.values, color=colors)
            ax1.set_xticks(range(len(regional_data)))
            ax1.set_xticklabels(regional_data.index, rotation=45, ha='right')
            ax1.set_ylabel(color_label)
            ax1.set_title(f'{title_suffix} по регионам\n({period}, {disease_filter})')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Добавление значений на столбцы
            for i, (bar, value) in enumerate(zip(bars, regional_data.values)):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                        f'{value:,.1f}', ha='center', va='bottom', fontsize=9, rotation=0)
            
            # График 2: Круговая диаграмма топ-8 регионов
            top_regions = regional_data.head(8)
            others = regional_data[8:].sum() if len(regional_data) > 8 else 0
            
            if others > 0:
                plot_data = pd.concat([top_regions, pd.Series([others], index=['Другие'])])
            else:
                plot_data = top_regions
            
            if len(plot_data) > 0:
                wedges, texts, autotexts = ax2.pie(plot_data.values, labels=plot_data.index, 
                                                  autopct='%1.1f%%', startangle=90)
                ax2.set_title(f'Распределение по регионам\n({metric}, {period})')
                
                # Улучшаем вид текста
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(8)
            else:
                ax2.text(0.5, 0.5, 'Нет данных\nдля отображения', 
                        ha='center', va='center', transform=ax2.transAxes)
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.map_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Обновление статистики
            if len(regional_data) > 0:
                stats_text = f"""СТАТИСТИКА ПО КАРТЕ ({period})
══════════════════════════════════════
Показатель: {metric}
Заболевание: {disease_filter}
Всего регионов: {len(regional_data)}

🏆 ТОП-3:
🥇 {regional_data.index[0]}: {regional_data.iloc[0]:,.1f}"""
                
                if len(regional_data) > 1:
                    stats_text += f"\n🥈 {regional_data.index[1]}: {regional_data.iloc[1]:,.1f}"
                if len(regional_data) > 2:
                    stats_text += f"\n🥉 {regional_data.index[2]}: {regional_data.iloc[2]:,.1f}"
                    
                stats_text += f"""

📊 Статистика:
• Среднее: {regional_data.mean():,.1f}
• Медиана: {regional_data.median():,.1f}
• Макс/мин: {regional_data.max():,.1f} / {regional_data.min():,.1f}"""
            else:
                stats_text = "Нет данных для отображения статистики"
            
            self.map_stats_text.delete(1.0, tk.END)
            self.map_stats_text.insert(1.0, stats_text)
            
            self.update_status(f"Региональная карта построена: {metric} за {period}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении региональной карты: {str(e)}")
        
    def build_density_map(self):
        """Построение карты плотности (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        try:
            # Создание scatter plot для имитации плотности
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
            
            # Подготовка данных
            data = self.current_data.copy()
            period = self.map_period.get()
            disease_filter = self.map_disease.get()
            
            # Применяем фильтры
            if disease_filter != 'Все' and 'Заболевание' in data.columns:
                data = data[data['Заболевание'] == disease_filter]
            
            if period != 'Все годы':
                try:
                    data['Год'] = pd.to_datetime(data['Дата']).dt.year
                    data = data[data['Год'] == int(period)]
                except:
                    pass
            
            regional_data = data.groupby('Регион').agg({
                'Количество': 'sum'
            }).reset_index()
            
            if len(regional_data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для построения карты плотности")
                return
            
            # Добавляем средний возраст если есть колонка Возраст
            if 'Возраст' in data.columns:
                avg_age = data.groupby('Регион')['Возраст'].mean()
                regional_data = regional_data.merge(avg_age.reset_index(), on='Регион', how='left')
                color_data = regional_data['Возраст'].fillna(regional_data['Возраст'].mean())
                color_label = 'Средний возраст'
            else:
                color_data = regional_data['Количество']
                color_label = 'Количество случаев'
            
            # График 1: Scatter plot с координатами
            # Создание условных координат для регионов
            np.random.seed(42)
            x_coords = np.random.uniform(0, 10, len(regional_data))
            y_coords = np.random.uniform(0, 10, len(regional_data))
            
            # Размер точек пропорционален количеству случаев
            max_cases = regional_data['Количество'].max()
            sizes = (regional_data['Количество'] / max_cases * 800) + 100
            
            scatter = ax1.scatter(x_coords, y_coords, 
                            s=sizes, 
                            c=color_data, 
                            cmap='viridis', alpha=0.7, edgecolors='black', linewidth=1)
            
            # Подписи регионов
            for i, region in enumerate(regional_data['Регион']):
                ax1.annotate(region, (x_coords[i], y_coords[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            ax1.set_xlabel('Условная долгота')
            ax1.set_ylabel('Условная широта')
            ax1.set_title(f'Карта плотности заболеваемости\n({period}, {disease_filter})\nРазмер = количество случаев')
            ax1.grid(True, alpha=0.3)
            
            # Цветовая шкала
            plt.colorbar(scatter, ax=ax1, label=color_label)
            
            # График 2: Гистограмма распределения
            ax2.hist(regional_data['Количество'], bins=min(10, len(regional_data)), 
                    color='skyblue', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Количество случаев')
            ax2.set_ylabel('Количество регионов')
            ax2.set_title(f'Распределение плотности\nпо регионам ({period})')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.map_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Статистика
            stats_text = f"""КАРТА ПЛОТНОСТИ ({period})
══════════════════════════════════════
Заболевание: {disease_filter}
Регионов: {len(regional_data)}
Общее количество случаев: {regional_data['Количество'].sum():,}

📊 Распределение плотности:
• Максимальная: {regional_data['Количество'].max():,}
• Минимальная: {regional_data['Количество'].min():,}
• Средняя: {regional_data['Количество'].mean():,.1f}
• Медиана: {regional_data['Количество'].median():,.1f}

🎯 Коэффициент вариации: {(regional_data['Количество'].std() / regional_data['Количество'].mean() * 100):,.1f}%"""
            
            self.map_stats_text.delete(1.0, tk.END)
            self.map_stats_text.insert(1.0, stats_text)
            
            self.update_status(f"Карта плотности построена для {period}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении карты плотности: {str(e)}")

    def build_temporal_map(self):
        """Построение временной карты (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        try:
            # Создание временной карты
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Подготовка данных по годам
            data = self.current_data.copy()
            disease_filter = self.map_disease.get()
            
            # Фильтр по заболеванию
            if disease_filter != 'Все' and 'Заболевание' in data.columns:
                data = data[data['Заболевание'] == disease_filter]
            
            data['Дата'] = pd.to_datetime(data['Дата'])
            data['Год'] = data['Дата'].dt.year
            data['Месяц'] = data['Дата'].dt.month
            
            # График 1: Тепловая карта год-регион
            try:
                temporal_data = data.pivot_table(index='Год', columns='Регион', 
                                            values='Количество', aggfunc='sum', fill_value=0)
                
                if len(temporal_data.index) > 0 and len(temporal_data.columns) > 0:
                    # Берем топ-12 регионов для читаемости
                    top_regions = temporal_data.sum(axis=0).sort_values(ascending=False).head(12).index
                    temporal_subset = temporal_data[top_regions]
                    
                    im1 = ax1.imshow(temporal_subset.values, cmap='YlOrRd', aspect='auto')
                    ax1.set_xticks(range(len(temporal_subset.columns)))
                    ax1.set_yticks(range(len(temporal_subset.index)))
                    ax1.set_xticklabels(temporal_subset.columns, rotation=45, ha='right')
                    ax1.set_yticklabels(temporal_subset.index)
                    ax1.set_xlabel('Регион (топ-12)')
                    ax1.set_ylabel('Год')
                    ax1.set_title(f'Временная динамика по регионам\n({disease_filter})')
                    
                    plt.colorbar(im1, ax=ax1, label='Количество случаев')
                    
                    # Добавляем значения на ячейки для лучшей читаемости
                    for i in range(len(temporal_subset.index)):
                        for j in range(len(temporal_subset.columns)):
                            value = temporal_subset.iloc[i, j]
                            if value > 0:
                                text_color = 'white' if value > temporal_subset.values.max() * 0.5 else 'black'
                                ax1.text(j, i, f'{int(value)}', ha='center', va='center', 
                                        color=text_color, fontsize=8, fontweight='bold')
                else:
                    ax1.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center', transform=ax1.transAxes)
                    ax1.set_title('Временная динамика по регионам (нет данных)')
            except Exception as e:
                ax1.text(0.5, 0.5, f'Ошибка: {str(e)[:50]}...', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Ошибка временной динамики')
            
            # График 2: Тепловая карта месяц-год
            try:
                monthly_data = data.pivot_table(index='Месяц', columns='Год', 
                                              values='Количество', aggfunc='sum', fill_value=0)
                
                if len(monthly_data.index) > 0 and len(monthly_data.columns) > 0:
                    im2 = ax2.imshow(monthly_data.values, cmap='RdYlBu_r', aspect='auto')
                    ax2.set_xticks(range(len(monthly_data.columns)))
                    ax2.set_yticks(range(len(monthly_data.index)))
                    ax2.set_xticklabels(monthly_data.columns)
                    ax2.set_yticklabels([f'Мес.{m}' for m in monthly_data.index])
                    ax2.set_xlabel('Год')
                    ax2.set_ylabel('Месяц')
                    ax2.set_title(f'Сезонная динамика по годам\n({disease_filter})')
                    
                    plt.colorbar(im2, ax=ax2, label='Количество случаев')
                    
                    # Добавляем значения на ячейки
                    for i in range(len(monthly_data.index)):
                        for j in range(len(monthly_data.columns)):
                            value = monthly_data.iloc[i, j]
                            if value > 0:
                                text_color = 'white' if value > monthly_data.values.max() * 0.5 else 'black'
                                ax2.text(j, i, f'{int(value)}', ha='center', va='center', 
                                        color=text_color, fontsize=8, fontweight='bold')
                else:
                    ax2.text(0.5, 0.5, 'Недостаточно данных', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Сезонная динамика (нет данных)')
            except Exception as e:
                ax2.text(0.5, 0.5, f'Ошибка: {str(e)[:50]}...', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Ошибка сезонной динамики')
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.map_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Статистика
            years = sorted(data['Год'].unique()) if 'Год' in data.columns else []
            total_records = len(data)
            
            stats_text = f"""ВРЕМЕННАЯ КАРТА
══════════════════════════════════════
Заболевание: {disease_filter}
Период: {min(years) if years else 'Н/Д'} - {max(years) if years else 'Н/Д'}
Количество лет: {len(years)}
Всего записей: {total_records:,}

📈 Динамика:
• Среднее в год: {total_records / len(years) if years else 0:,.0f}
• Всего случаев: {data['Количество'].sum() if 'Количество' in data.columns else 0:,}

🗓️ Покрытие данных:
• Месяцев с данными: {data['Месяц'].nunique() if 'Месяц' in data.columns else 0}
• Регионов охвачено: {data['Регион'].nunique() if 'Регион' in data.columns else 0}"""
            
            self.map_stats_text.delete(1.0, tk.END)
            self.map_stats_text.insert(1.0, stats_text)
            
            self.update_status(f"Временная карта построена для {disease_filter}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении временной карты: {str(e)}")

    def build_kz_cartogram(self):
        """Картограмма регионов Казахстана по выбранному показателю"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            data = self.current_data.copy()
            metric = self.map_metric.get()
            period = self.map_period.get()
            disease_filter = self.map_disease.get()

            if disease_filter != 'Все' and 'Заболевание' in data.columns:
                data = data[data['Заболевание'] == disease_filter]

            if period != 'Все годы':
                data['Год'] = pd.to_datetime(data['Дата']).dt.year
                data = data[data['Год'] == int(period)]

            if len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return

            if metric == 'Темп роста':
                data['Год'] = pd.to_datetime(data['Дата']).dt.year
                yearly = data.groupby(['Регион', 'Год'])['Количество'].sum().unstack(fill_value=0)
                if yearly.shape[1] >= 2:
                    last_year = yearly.columns[-1]
                    prev_year = yearly.columns[-2]
                    values = ((yearly[last_year] - yearly[prev_year]) / yearly[prev_year].replace(0, 1) * 100)
                    color_label = 'Темп роста (%)'
                else:
                    values = data.groupby('Регион')['Количество'].sum()
                    color_label = 'Количество случаев'
            elif metric == 'Всего случаев':
                values = data.groupby('Регион')['Количество'].sum()
                color_label = 'Количество случаев'
            else:
                values = data.groupby('Регион')['Количество'].mean()
                color_label = metric

            with open('kazakhstan_regions.json', 'r', encoding='utf-8') as f:
                coords = json.load(f)['regions']

            lons = [coords[r]['lon'] for r in values.index if r in coords]
            lats = [coords[r]['lat'] for r in values.index if r in coords]
            vals = [values[r] for r in values.index if r in coords]

            sc = ax.scatter(lons, lats, c=vals, cmap='coolwarm', s=300, edgecolors='black')

            for region in values.index:
                if region in coords:
                    ax.text(coords[region]['lon'], coords[region]['lat'], region, ha='center', va='center', fontsize=8)

            plt.colorbar(sc, ax=ax, label=color_label)
            ax.set_title(f'{metric} по регионам Казахстана ({period})')
            ax.set_xlabel('Долгота')
            ax.set_ylabel('Широта')

            for widget in self.map_plot_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.map_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            self.map_stats_text.delete(1.0, tk.END)
            self.map_stats_text.insert(1.0, f'Регионов: {len(values)}\nСреднее значение: {np.mean(vals):.1f}')
            self.update_status('Карта Казахстана построена')

        except Exception as e:
            messagebox.showerror('Ошибка', f'Ошибка построения карты Казахстана: {str(e)}')
            
    def perform_analysis(self):
        """Выполнение выбранного типа анализа"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        analysis_type = self.analysis_type.get()
        
        try:
            # Очищаем предыдущие результаты
            for widget in self.analysis_plot_frame.winfo_children():
                widget.destroy()
            self.analysis_canvas = None
            
            if analysis_type == "seasonality":
                self.analyze_seasonality()
            elif analysis_type == "regions":
                self.analyze_regions()
            elif analysis_type == "age_groups":
                self.analyze_age_groups()
            elif analysis_type == "correlation":
                self.analyze_correlation()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выполнении анализа: {str(e)}")

    def save_analysis_plot(self):
        """Сохранение текущего графика анализа"""
        try:
            canvas_widget = getattr(self, 'analysis_canvas', None)
            if canvas_widget is None:
                messagebox.showwarning("Предупреждение", "Нет графика для сохранения!")
                return
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG файлы", "*.png"),
                    ("PDF файлы", "*.pdf"),
                    ("SVG файлы", "*.svg"),
                    ("JPEG файлы", "*.jpg")
                ],
                title="Сохранить график"
            )
            
            if filename:
                canvas_widget.figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Успех", f"График сохранен: {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении графика: {str(e)}")

    def quick_analysis(self):
        """Быстрый анализ данных"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Переключение на вкладку анализа
        self.notebook.select(1)
        # Выполнение анализа сезонности по умолчанию
        self.analyze_seasonality()
        
    def quick_forecast(self):
        """Быстрое прогнозирование"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Переключение на вкладку прогнозов
        self.notebook.select(2)
        # Выполнение прогноза
        self.build_forecast()
        
    def generate_report(self):
        """Быстрая генерация отчета"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        # Переключение на вкладку отчетов
        self.notebook.select(3)
        # Создание сводного отчета
        self.create_report()
        
    def analyze_seasonality(self):
        """Анализ сезонности заболеваний"""
        try:
            # Создание фигуры с множественными графиками
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Подготовка данных
            data = self.get_analysis_filtered_data()
            if data is None or len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return
            data['Дата'] = pd.to_datetime(data['Дата'])
            data['Месяц'] = data['Дата'].dt.month
            data['Год'] = data['Дата'].dt.year
            data['Сезон'] = data['Месяц'].map({
                12: 'Зима', 1: 'Зима', 2: 'Зима',
                3: 'Весна', 4: 'Весна', 5: 'Весна',
                6: 'Лето', 7: 'Лето', 8: 'Лето',
                9: 'Осень', 10: 'Осень', 11: 'Осень'
            })
            
            # График 1: Общая сезонность по месяцам
            monthly_data = data.groupby('Месяц')['Количество'].sum()
            ax1.plot(monthly_data.index, monthly_data.values, marker='o', linewidth=3, markersize=8, color='#2E86AB')
            ax1.fill_between(monthly_data.index, monthly_data.values, alpha=0.3, color='#2E86AB')
            ax1.set_xlabel('Месяц')
            ax1.set_ylabel('Количество случаев')
            ax1.set_title('Общая сезонность заболеваемости', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(range(1, 13))
            
            # График 2: Сезонность по заболеваниям (топ-5)
            if 'Заболевание' in data.columns:
                diseases = data.groupby('Заболевание')['Количество'].sum().sort_values(ascending=False).head(5).index
                colors = plt.cm.Set1(np.linspace(0, 1, len(diseases)))
                
                for disease, color in zip(diseases, colors):
                    disease_data = data[data['Заболевание'] == disease]
                    monthly_disease = disease_data.groupby('Месяц')['Количество'].sum()
                    ax2.plot(monthly_disease.index, monthly_disease.values, 
                            marker='o', label=disease, linewidth=2, color=color)
                
                ax2.set_xlabel('Месяц')
                ax2.set_ylabel('Количество случаев')
                ax2.set_title('Сезонность по типам заболеваний', fontsize=14, fontweight='bold')
                ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax2.grid(True, alpha=0.3)
                ax2.set_xticks(range(1, 13))
            
            # График 3: Распределение по сезонам
            seasonal_data = data.groupby('Сезон')['Количество'].sum()
            colors = ['lightblue', 'lightgreen', 'orange', 'lightcoral']
            wedges, texts, autotexts = ax3.pie(seasonal_data.values, labels=seasonal_data.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
            ax3.set_title('Распределение по сезонам', fontsize=14, fontweight='bold')
            
            # Улучшаем вид текста на диаграмме
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # График 4: Динамика по годам
            yearly_monthly = data.groupby(['Год', 'Месяц'])['Количество'].sum().unstack(fill_value=0)
            
            if len(yearly_monthly.index) > 1:
                for year in yearly_monthly.index:
                    ax4.plot(range(1, 13), yearly_monthly.loc[year], marker='o', label=str(year), linewidth=2)
                ax4.set_xlabel('Месяц')
                ax4.set_ylabel('Количество случаев')
                ax4.set_title('Сравнение сезонности по годам', fontsize=14, fontweight='bold')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                ax4.set_xticks(range(1, 13))
            else:
                ax4.text(0.5, 0.5, 'Недостаточно данных\nдля сравнения по годам', 
                        ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            
            plt.tight_layout()
            
            # Встраивание графика в интерфейс
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.analysis_canvas = canvas
            self.analysis_canvas = canvas
            self.analysis_canvas = canvas
            self.analysis_canvas = canvas
            
            self.update_status("Анализ сезонности выполнен успешно")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе сезонности: {str(e)}")

    def analyze_regions(self):
        """Анализ заболеваемости по регионам"""
        try:
            # Создание фигуры
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Подготовка данных
            data = self.get_analysis_filtered_data()
            if data is None or len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return
            regional_data = data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
            
            # График 1: Столбчатая диаграмма по регионам
            colors = plt.cm.viridis(np.linspace(0, 1, len(regional_data)))
            bars = ax1.bar(range(len(regional_data)), regional_data.values, color=colors)
            ax1.set_xticks(range(len(regional_data)))
            ax1.set_xticklabels(regional_data.index, rotation=45, ha='right')
            ax1.set_xlabel('Регион')
            ax1.set_ylabel('Количество случаев')
            ax1.set_title('Заболеваемость по регионам', fontsize=14, fontweight='bold')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Добавление значений на столбцы
            for bar, value in zip(bars, regional_data.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + value*0.01,
                        f'{value:,.0f}', ha='center', va='bottom', fontsize=8)
            
            # График 2: Круговая диаграмма топ-регионов
            top_regions = regional_data.head(8)
            others = regional_data[8:].sum()
            if others > 0:
                plot_data = pd.concat([top_regions, pd.Series([others], index=['Другие'])])
            else:
                plot_data = top_regions
                
            ax2.pie(plot_data.values, labels=plot_data.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Доля регионов в общей заболеваемости', fontsize=14, fontweight='bold')
            
            # График 3: Тепловая карта регион-заболевание
            if 'Заболевание' in data.columns:
                try:
                    heatmap_data = data.pivot_table(index='Регион', columns='Заболевание', 
                                                  values='Количество', aggfunc='sum', fill_value=0)
                    
                    # Берем топ-10 регионов и топ-5 заболеваний для читаемости
                    top_regions_heat = heatmap_data.sum(axis=1).sort_values(ascending=False).head(10).index
                    top_diseases_heat = heatmap_data.sum(axis=0).sort_values(ascending=False).head(5).index
                    
                    heatmap_subset = heatmap_data.loc[top_regions_heat, top_diseases_heat]
                    
                    im = ax3.imshow(heatmap_subset.values, cmap='YlOrRd', aspect='auto')
                    ax3.set_xticks(range(len(heatmap_subset.columns)))
                    ax3.set_yticks(range(len(heatmap_subset.index)))
                    ax3.set_xticklabels(heatmap_subset.columns, rotation=45, ha='right')
                    ax3.set_yticklabels(heatmap_subset.index)
                    ax3.set_title('Тепловая карта: Топ регионы × заболевания', fontsize=14, fontweight='bold')
                    
                    # Цветовая шкала
                    plt.colorbar(im, ax=ax3, label='Количество случаев')
                    
                except Exception as e:
                    ax3.text(0.5, 0.5, f'Ошибка построения\nтепловой карты:\n{str(e)}', 
                            ha='center', va='center', transform=ax3.transAxes)
            
            # График 4: Статистика по регионам
            regional_stats = data.groupby('Регион')['Количество'].agg(['sum', 'mean', 'std']).fillna(0)
            regional_stats = regional_stats.sort_values('sum', ascending=True).tail(10)  # Топ-10
            
            ax4.barh(range(len(regional_stats)), regional_stats['sum'], color='lightcoral', alpha=0.7, label='Всего')
            ax4.set_yticks(range(len(regional_stats)))
            ax4.set_yticklabels(regional_stats.index)
            ax4.set_xlabel('Количество случаев')
            ax4.set_title('Топ-10 регионов (горизонтальная диаграмма)', fontsize=14, fontweight='bold')
            ax4.grid(True, axis='x', alpha=0.3)
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.update_status("Анализ по регионам выполнен успешно")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе по регионам: {str(e)}")

    def analyze_age_groups(self):
        """Анализ заболеваемости по возрастным группам"""
        try:
            if 'Возраст' not in self.current_data.columns:
                messagebox.showwarning("Предупреждение", "В данных отсутствует колонка 'Возраст'")
                return
            
            # Создание фигуры
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Подготовка данных
            data = self.get_analysis_filtered_data()
            if data is None or len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return
            data = data.dropna(subset=['Возраст'])  # Убираем записи без возраста
            
            # Создание возрастных групп
            data['Возрастная группа'] = pd.cut(data['Возраст'], 
                                              bins=[0, 14, 30, 45, 60, 100],
                                              labels=['0-14', '15-30', '31-45', '46-60', '60+'])
            
            # График 1: Распределение по возрастным группам
            age_data = data.groupby('Возрастная группа')['Количество'].sum()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            bars = ax1.bar(age_data.index, age_data.values, color=colors[:len(age_data)])
            ax1.set_xlabel('Возрастная группа')
            ax1.set_ylabel('Количество случаев')
            ax1.set_title('Заболеваемость по возрастным группам', fontsize=14, fontweight='bold')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # Добавление значений на столбцы
            for bar, value in zip(bars, age_data.values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + value*0.01,
                        f'{value:,.0f}', ha='center', va='bottom', fontsize=10)
            
            # График 2: Средний возраст по заболеваниям
            if 'Заболевание' in data.columns:
                disease_age = data.groupby('Заболевание')['Возраст'].mean().sort_values()
                ax2.barh(disease_age.index, disease_age.values, color='lightgreen')
                ax2.set_xlabel('Средний возраст')
                ax2.set_ylabel('Заболевание')
                ax2.set_title('Средний возраст пациентов по заболеваниям', fontsize=14, fontweight='bold')
                ax2.grid(True, axis='x', alpha=0.3)
            
            # График 3: Гистограмма возрастов
            ax3.hist(data['Возраст'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Возраст')
            ax3.set_ylabel('Частота')
            ax3.set_title('Распределение возрастов пациентов', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # График 4: Возрастные группы по полу (если есть данные)
            if 'Пол' in data.columns:
                try:
                    gender_age = data.groupby(['Возрастная группа', 'Пол'])['Количество'].sum().unstack(fill_value=0)
                    gender_age.plot(kind='bar', ax=ax4, color=['lightblue', 'lightpink'])
                    ax4.set_xlabel('Возрастная группа')
                    ax4.set_ylabel('Количество случаев')
                    ax4.set_title('Заболеваемость по возрасту и полу', fontsize=14, fontweight='bold')
                    ax4.legend(title='Пол')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, axis='y', alpha=0.3)
                except:
                    ax4.text(0.5, 0.5, 'Недостаточно данных\nпо полу', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                # Box plot возрастов по заболеваниям
                if 'Заболевание' in data.columns:
                    diseases_for_box = data['Заболевание'].value_counts().head(5).index
                    box_data = [data[data['Заболевание'] == disease]['Возраст'].values 
                               for disease in diseases_for_box]
                    ax4.boxplot(box_data, labels=diseases_for_box)
                    ax4.set_xlabel('Заболевание')
                    ax4.set_ylabel('Возраст')
                    ax4.set_title('Распределение возрастов по заболеваниям', fontsize=14, fontweight='bold')
                    ax4.tick_params(axis='x', rotation=45)
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.update_status("Анализ по возрастным группам выполнен успешно")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе по возрастным группам: {str(e)}")

    def analyze_correlation(self):
        """Анализ корреляций между факторами"""
        try:
            # Создание фигуры
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Подготовка данных для корреляционной матрицы
            data = self.get_analysis_filtered_data()
            if data is None or len(data) == 0:
                messagebox.showwarning("Предупреждение", "Нет данных для выбранных фильтров")
                return
            
            # Выбираем только числовые колонки
            numeric_data = data.select_dtypes(include=[np.number])
            
            if len(numeric_data.columns) < 2:
                messagebox.showwarning("Предупреждение", "Недостаточно числовых данных для анализа корреляций")
                return
            
            # График 1: Корреляционная матрица
            corr_matrix = numeric_data.corr()
            
            im = ax1.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax1.set_xticks(range(len(corr_matrix.columns)))
            ax1.set_yticks(range(len(corr_matrix.index)))
            ax1.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
            ax1.set_yticklabels(corr_matrix.index)
            ax1.set_title('Корреляционная матрица', fontsize=14, fontweight='bold')
            
            # Добавляем значения корреляций на график
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                   ha="center", va="center", 
                                   color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white")
            
            plt.colorbar(im, ax=ax1, label='Коэффициент корреляции')
            
            # График 2: Scatter plot самых коррелированных переменных
            if len(numeric_data.columns) >= 2:
                # Находим пару с наибольшей корреляцией (исключая диагональ)
                corr_abs = np.abs(corr_matrix.values)
                np.fill_diagonal(corr_abs, 0)
                max_corr_idx = np.unravel_index(np.argmax(corr_abs), corr_abs.shape)
                
                var1, var2 = corr_matrix.columns[max_corr_idx[1]], corr_matrix.index[max_corr_idx[0]]
                
                ax2.scatter(numeric_data[var1], numeric_data[var2], alpha=0.6, color='steelblue')
                ax2.set_xlabel(var1)
                ax2.set_ylabel(var2)
                ax2.set_title(f'Scatter plot: {var1} vs {var2}\nКорреляция: {corr_matrix.loc[var2, var1]:.3f}', 
                             fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Добавляем линию тренда
                z = np.polyfit(numeric_data[var1].dropna(), numeric_data[var2].dropna(), 1)
                p = np.poly1d(z)
                ax2.plot(numeric_data[var1], p(numeric_data[var1]), "r--", alpha=0.8)
            
            # График 3: Распределение корреляций
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            ax3.hist(corr_values, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
            ax3.set_xlabel('Коэффициент корреляции')
            ax3.set_ylabel('Частота')
            ax3.set_title('Распределение коэффициентов корреляции', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # График 4: Тепловая карта временных корреляций
            if 'Дата' in data.columns:
                try:
                    data['Дата_dt'] = pd.to_datetime(data['Дата'])
                    data['Месяц'] = data['Дата_dt'].dt.month
                    
                    # Корреляция по месяцам
                    monthly_corr = []
                    months = []
                    
                    for month in range(1, 13):
                        month_data = data[data['Месяц'] == month]
                        if len(month_data) > 5:  # Минимум данных для корреляции
                            month_numeric = month_data.select_dtypes(include=[np.number])
                            if len(month_numeric.columns) >= 2:
                                corr = month_numeric.corr().iloc[0, 1] if len(month_numeric.columns) == 2 else month_numeric.corr().values[0, 1]
                                monthly_corr.append(corr)
                                months.append(month)
                    
                    if monthly_corr:
                        ax4.plot(months, monthly_corr, marker='o', linewidth=2, markersize=8, color='green')
                        ax4.set_xlabel('Месяц')
                        ax4.set_ylabel('Корреляция')
                        ax4.set_title('Сезонная динамика корреляций', fontsize=14, fontweight='bold')
                        ax4.grid(True, alpha=0.3)
                        ax4.set_xticks(range(1, 13))
                    else:
                        ax4.text(0.5, 0.5, 'Недостаточно данных\nдля временного анализа', 
                                ha='center', va='center', transform=ax4.transAxes)
                except:
                    ax4.text(0.5, 0.5, 'Ошибка временного\nанализа', 
                            ha='center', va='center', transform=ax4.transAxes)
            else:
                ax4.text(0.5, 0.5, 'Колонка "Дата"\nне найдена', 
                        ha='center', va='center', transform=ax4.transAxes)
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.analysis_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            self.update_status("Анализ корреляций выполнен успешно")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при анализе корреляций: {str(e)}")
                    
    def build_forecast(self):
        """Построение прогноза заболеваемости"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные!")
            return
            
        model_type = self.model_var.get()
        periods = self.forecast_period.get()
        
        # Очистка области графиков
        for widget in self.forecast_plot_frame.winfo_children():
            widget.destroy()
            
        self.update_status(f"Строится прогноз {model_type}...")
        self.root.update()  # Обновляем интерфейс
            
        try:
            # Создание прогноза в зависимости от выбранной модели
            if model_type == "XGBoost":
                # Проверяем глобальную переменную
                if globals().get('XGBOOST_AVAILABLE', False):
                    self.forecast_xgboost()
                else:
                    # Показываем информативное сообщение
                    messagebox.showinfo("XGBoost недоступен", 
                                    "XGBoost не установлен или не может быть загружен.\n\n"
                                    "Статус установки библиотек:\n"
                                    f"• XGBoost: {globals().get('XGBOOST_AVAILABLE', False)}\n"
                                    f"• scikit-learn: {globals().get('SKLEARN_AVAILABLE', False)}\n"
                                    f"• statsmodels: {globals().get('STATSMODELS_AVAILABLE', False)}\n\n"
                                    "Попробуйте перезапустить программу после установки библиотек.")
                    # Переключаемся на SARIMA как запасной вариант
                    self.model_var.set("SARIMA")
                    self.forecast_sarima()
                    
            elif model_type == "Random Forest":
                if globals().get('SKLEARN_AVAILABLE', False):
                    self.forecast_ml()
                else:
                    messagebox.showerror("Ошибка", "scikit-learn не установлен!\nПереключение на SARIMA.")
                    self.model_var.set("SARIMA")
                    self.forecast_sarima()
                    
            elif model_type == "Linear Regression":
                if globals().get('SKLEARN_AVAILABLE', False):
                    self.forecast_linear_regression()
                else:
                    messagebox.showerror("Ошибка", "scikit-learn не установлен!\nПереключение на SARIMA.")
                    self.model_var.set("SARIMA")
                    self.forecast_sarima()
            else:
                # SARIMA по умолчанию - всегда доступен
                self.forecast_sarima()
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении прогноза {model_type}: {str(e)}")
            # Показываем traceback для отладки
            import traceback
            print(f"Полная ошибка в {model_type}:")
            traceback.print_exc()
            
            # Пробуем запасной вариант - SARIMA
            try:
                print("Попытка использовать SARIMA как запасной вариант...")
                self.model_var.set("SARIMA")
                self.forecast_sarima()
            except Exception as fallback_error:
                messagebox.showerror("Критическая ошибка", 
                                f"Не удалось построить ни один прогноз:\n{str(fallback_error)}")
                print(f"Ошибка в запасном SARIMA: {fallback_error}")
                
            self.update_status("Ошибка при построении прогноза")

    def forecast_sarima(self):
        """Настоящее прогнозирование SARIMA"""
        try:
            # Подготовка данных
            data = self.current_data.copy()
            region_filter = getattr(self, 'forecast_region_var', None)
            if region_filter and region_filter != 'Все' and 'Регион' in data.columns:
                data = data[data['Регион'] == region_filter]
            data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')
            data = data.dropna(subset=['Дата'])
            monthly_data = data.resample('MS', on='Дата')['Количество'].sum()

            if (monthly_data > 0).sum() < 24:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для SARIMA (нужно минимум 24 месяца)")
                return

            monthly_data = monthly_data[monthly_data > 0]
            
            # Проверка стационарности
            def check_stationarity(timeseries):
                """Проверка стационарности временного ряда"""
                if STATSMODELS_AVAILABLE:
                    result = adfuller(timeseries)
                    return result[1] < 0.05  # p-value < 0.05 означает стационарность
                return True  # Предполагаем стационарность, если statsmodels недоступен
            
            # Приведение к стационарному виду
            diff_data = monthly_data
            diff_order = 0
            
            if STATSMODELS_AVAILABLE:
                while not check_stationarity(diff_data.dropna()) and diff_order < 2:
                    diff_data = diff_data.diff()
                    diff_order += 1
            
            # Простая реализация ARIMA без statsmodels
            if not STATSMODELS_AVAILABLE:
                # Упрощенная модель: экспоненциальное сглаживание с трендом и сезонностью
                periods = self.forecast_period.get()
                
                # Вычисляем тренд
                X = np.arange(len(monthly_data))
                trend_coef = np.polyfit(X, monthly_data.values, 1)
                
                # Вычисляем сезонную компоненту (12-месячная)
                seasonal_component = np.zeros(12)
                for i in range(12):
                    month_values = [monthly_data.iloc[j] for j in range(i, len(monthly_data), 12)]
                    if month_values:
                        seasonal_component[i] = np.mean(month_values) - monthly_data.mean()
                
                # Прогноз
                last_value = monthly_data.iloc[-1]
                last_date = monthly_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                            periods=periods, freq='M')
                forecast_values = []
                
                for i in range(periods):
                    # Тренд
                    trend_value = trend_coef[0] * (len(monthly_data) + i) + trend_coef[1]
                    
                    # Сезонность
                    month_idx = (last_date.month + i) % 12
                    seasonal_value = seasonal_component[month_idx]
                    
                    # Объединяем компоненты
                    forecast_val = max(0, trend_value + seasonal_value)
                    forecast_values.append(forecast_val)
                
                method_name = "Упрощенная SARIMA (тренд + сезонность)"
                
            else:
                # Используем настоящую ARIMA из statsmodels
                try:
                    # Автоматический подбор параметров ARIMA
                    best_aic = float('inf')
                    best_model = None
                    best_params = None
                    
                    # Простой перебор параметров
                    for p in range(0, 3):
                        for d in range(0, 2):
                            for q in range(0, 3):
                                try:
                                    model = ARIMA(monthly_data, order=(p, d, q))
                                    fitted_model = model.fit()
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_model = fitted_model
                                        best_params = (p, d, q)
                                except:
                                    continue
                    
                    if best_model is None:
                        raise ValueError("Не удалось подобрать подходящие параметры ARIMA")
                    
                    # Прогноз
                    periods = self.forecast_period.get()
                    forecast_result = best_model.forecast(steps=periods)
                    forecast_values = np.maximum(forecast_result, 0)  # Неотрицательные значения
                    
                    last_date = monthly_data.index[-1]
                    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                                periods=periods, freq='M')
                    
                    method_name = f"ARIMA{best_params} (AIC: {best_aic:.1f})"
                    
                except Exception as e:
                    # Fallback к упрощенной модели
                    print(f"Ошибка ARIMA: {e}, используем упрощенную модель")
                    return self.forecast_sarima()  # Рекурсивно с STATSMODELS_AVAILABLE = False
            
            # Создание графика
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # График 1: Прогноз
            ax1.plot(monthly_data.index, monthly_data.values, 
                    label='Исторические данные', marker='o', linewidth=2, color='blue')
            ax1.plot(forecast_dates, forecast_values, 
                    label=f'Прогноз ({method_name})', color='red', marker='s', linestyle='--', linewidth=2)
            
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Количество случаев')
            ax1.set_title(f'SARIMA прогноз на {periods} месяцев\n{method_name}', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Декомпозиция временного ряда
            if STATSMODELS_AVAILABLE and len(monthly_data) >= 24:
                try:
                    decomposition = seasonal_decompose(monthly_data, model='additive', period=12)
                    
                    ax2.plot(monthly_data.index, decomposition.trend.dropna(), 
                            label='Тренд', linewidth=2, color='green')
                    ax2.plot(monthly_data.index, decomposition.seasonal, 
                            label='Сезонность', linewidth=1, alpha=0.7, color='orange')
                    ax2.set_xlabel('Дата')
                    ax2.set_ylabel('Компоненты')
                    ax2.set_title('Декомпозиция временного ряда', fontsize=14)
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                except:
                    ax2.plot(monthly_data.index, monthly_data.values, color='blue')
                    ax2.set_title('Исходный временной ряд')
            else:
                # Показываем остатки простой модели
                if len(monthly_data) >= 12:
                    rolling_mean = monthly_data.rolling(window=12).mean()
                    residuals = monthly_data - rolling_mean
                    ax2.plot(monthly_data.index, residuals, color='gray', alpha=0.7)
                    ax2.axhline(y=0, color='red', linestyle='--')
                    ax2.set_title('Остатки (отклонения от скользящего среднего)')
                    ax2.set_ylabel('Остатки')
                else:
                    ax2.plot(monthly_data.index, monthly_data.values, color='blue')
                    ax2.set_title('Исходные данные')
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохранение результатов
            self.forecast_results = {
                'dates': forecast_dates,
                'values': forecast_values,
                'model': 'SARIMA',
                'method_details': method_name
            }
            
            self.update_status(f"SARIMA прогноз построен на {periods} месяцев")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении SARIMA прогноза: {str(e)}")

    def forecast_xgboost(self):
        """Прогнозирование с использованием XGBoost (ИСПРАВЛЕННАЯ ВЕРСИЯ с кодировкой)"""
        if not XGBOOST_AVAILABLE:
            messagebox.showerror("Ошибка", 
                            "Библиотека XGBoost не установлена!\n\n"
                            "Для установки выполните:\n"
                            "pip install xgboost")
            return
            
        try:
            # Подготовка данных
            data = self.current_data.copy()
            region_filter = getattr(self, 'forecast_region_var', None)
            if region_filter and region_filter != 'Все' and 'Регион' in data.columns:
                data = data[data['Регион'] == region_filter]
            data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')
            data = data.dropna(subset=['Дата'])

            if len(data) == 0:
                messagebox.showerror("Ошибка", "Нет корректных данных")
                return

            monthly_data = data.resample('MS', on='Дата')['Количество'].sum()

            if (monthly_data > 0).sum() < 12:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для XGBoost прогнозирования")
                return

            monthly_data = monthly_data.reset_index()
            monthly_data['Месяц'] = monthly_data['Дата'].dt.month
            monthly_data['Год'] = monthly_data['Дата'].dt.year
            monthly_data['Период'] = monthly_data['Год'] * 12 + monthly_data['Месяц']
            
            # Создание расширенных признаков
            monthly_data['Лаг_1'] = monthly_data['Количество'].shift(1)
            monthly_data['Лаг_2'] = monthly_data['Количество'].shift(2)
            monthly_data['Лаг_3'] = monthly_data['Количество'].shift(3)
            
            # Скользящие средние
            monthly_data['МА_3'] = monthly_data['Количество'].rolling(window=3, min_periods=1).mean()
            monthly_data['МА_6'] = monthly_data['Количество'].rolling(window=6, min_periods=1).mean()
            
            # Сезонные признаки
            monthly_data['Сезон_sin'] = np.sin(2 * np.pi * monthly_data['Месяц'] / 12)
            monthly_data['Сезон_cos'] = np.cos(2 * np.pi * monthly_data['Месяц'] / 12)
            
            # Тренд
            monthly_data['Тренд'] = range(len(monthly_data))
            
            # Убираем строки с NaN
            monthly_data = monthly_data.dropna()
            
            if len(monthly_data) < 8:
                messagebox.showwarning("Предупреждение", "Недостаточно данных после обработки для XGBoost модели")
                return
            
            # Подготовка признаков и целевой переменной
            feature_columns = ['Период', 'Месяц', 'Тренд', 'Лаг_1', 'Лаг_2', 'Лаг_3', 
                            'МА_3', 'МА_6', 'Сезон_sin', 'Сезон_cos']
            
            X = monthly_data[feature_columns].values
            y = monthly_data['Количество'].values
            
            # Разделение на обучающую и тестовую выборки
            test_size = min(0.2, 3 / len(X))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            
            # Обучение XGBoost модели
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                objective='reg:squarederror'
            )
            
            xgb_model.fit(X_train, y_train)
            
            # Оценка качества модели
            y_pred = xgb_model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Прогнозирование
            periods = self.forecast_period.get()
            forecast_values = []
            forecast_dates = []
            
            last_period = monthly_data['Период'].iloc[-1]
            
            for i in range(periods):
                new_period = last_period + i + 1
                new_year = new_period // 12
                new_month = new_period % 12
                if new_month == 0:
                    new_month = 12
                    new_year -= 1
                
                # Подготовка признаков
                new_trend = len(monthly_data) + i
                
                # Лаговые признаки
                if i == 0:
                    lag_1 = monthly_data['Количество'].iloc[-1]
                    lag_2 = monthly_data['Количество'].iloc[-2] if len(monthly_data) > 1 else lag_1
                    lag_3 = monthly_data['Количество'].iloc[-3] if len(monthly_data) > 2 else lag_1
                    ma_3 = monthly_data['Количество'].iloc[-3:].mean() if len(monthly_data) > 2 else lag_1
                    ma_6 = monthly_data['Количество'].iloc[-6:].mean() if len(monthly_data) > 5 else lag_1
                else:
                    lag_1 = forecast_values[i-1] if i > 0 else monthly_data['Количество'].iloc[-1]
                    lag_2 = forecast_values[i-2] if i > 1 else monthly_data['Количество'].iloc[-1]
                    lag_3 = forecast_values[i-3] if i > 2 else monthly_data['Количество'].iloc[-1]
                    
                    recent_values = list(monthly_data['Количество'].iloc[-3:]) + forecast_values[:i]
                    ma_3 = np.mean(recent_values[-3:])
                    ma_6 = np.mean(recent_values[-6:])
                
                new_features = [
                    new_period, new_month, new_trend, lag_1, lag_2, lag_3, ma_3, ma_6,
                    np.sin(2 * np.pi * new_month / 12), np.cos(2 * np.pi * new_month / 12)
                ]
                
                # Прогноз
                X_new = np.array([new_features])
                forecast_value = max(0, xgb_model.predict(X_new)[0])
                forecast_values.append(forecast_value)
                
                # ИСПРАВЛЕНИЕ: Правильное создание дат прогноза
                try:
                    forecast_date = pd.Timestamp(year=int(new_year), month=int(new_month), day=1)
                    forecast_dates.append(forecast_date)
                except (ValueError, OverflowError) as e:
                    # Если не можем создать дату, используем последнюю известную дату + offset
                    last_known_date = pd.Timestamp(year=int(monthly_data['Год'].iloc[-1]), 
                                                month=int(monthly_data['Месяц'].iloc[-1]), day=1)
                    forecast_date = last_known_date + pd.DateOffset(months=i+1)
                    forecast_dates.append(forecast_date)
            
            # ИСПРАВЛЕНИЕ КОДИРОВКИ: Настройка matplotlib для кириллицы
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Создание улучшенной визуализации
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # График 1: Прогноз (более чистый дизайн)
            try:
                historical_dates = []
                for _, row in monthly_data.iterrows():
                    try:
                        date = pd.Timestamp(year=int(row['Год']), month=int(row['Месяц']), day=1)
                        historical_dates.append(date)
                    except:
                        continue
                
                if len(historical_dates) == len(monthly_data):
                    # Исторические данные
                    ax1.plot(historical_dates, monthly_data['Количество'], 
                            label='Исторические данные', marker='o', linewidth=2.5, 
                            color='#2E86AB', markersize=6, alpha=0.8)
                    
                    # Прогноз
                    if forecast_dates:
                        ax1.plot(forecast_dates, forecast_values, 
                                label='Прогноз XGBoost', color='#E74C3C', marker='s', 
                                linestyle='--', linewidth=3, markersize=7, alpha=0.9)
                    
                    # Добавляем вертикальную линию разделения
                    if historical_dates and forecast_dates:
                        ax1.axvline(x=historical_dates[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
                        ax1.text(historical_dates[-1], ax1.get_ylim()[1]*0.9, ' Прогноз начинается здесь', 
                                rotation=90, verticalalignment='top', fontsize=10, color='gray')
                    
            except Exception as e:
                print(f"Ошибка создания дат: {e}")
                # Fallback к индексам
                ax1.plot(range(len(monthly_data)), monthly_data['Количество'], 
                        label='Исторические данные', marker='o', linewidth=2.5, color='#2E86AB')
                if forecast_values:
                    forecast_start = len(monthly_data)
                    ax1.plot(range(forecast_start, forecast_start + len(forecast_values)), forecast_values, 
                            label='Прогноз XGBoost', color='#E74C3C', marker='s', linestyle='--', linewidth=3)
            
            # Улучшенное оформление первого графика
            ax1.set_xlabel('Период', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Количество случаев', fontsize=12, fontweight='bold')
            ax1.set_title(f'XGBoost: Прогноз заболеваемости на {periods} месяцев\n'
                        f'Точность: R² = {r2:.3f} | Ошибка: MAE = {mae:.1f}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax1.set_facecolor('#FAFAFA')
            
            # График 2: Упрощённый анализ качества модели
            if len(X_test) > 0 and len(y_test) > 0:
                # Точность модели - более простой и понятный график
                ax2.scatter(y_test, y_pred, alpha=0.8, color='#3498DB', s=80, edgecolors='white', linewidth=1.5)
                
                # Идеальная линия
                min_val = min(min(y_test), min(y_pred)) * 0.95
                max_val = max(max(y_test), max(y_pred)) * 1.05
                ax2.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3, alpha=0.8, label='Идеальный прогноз')
                
                # Добавляем статистику в правый верхний угол
                stats_text = f'Качество модели:\n' \
                            f'R² = {r2:.3f}\n' \
                            f'MAE = {mae:.1f}\n' \
                            f'Тестовых точек: {len(y_test)}'
                
                ax2.text(0.98, 0.02, stats_text, transform=ax2.transAxes, fontsize=11,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
                
                ax2.set_xlabel('Фактические значения', fontsize=12, fontweight='bold')
                ax2.set_ylabel('Предсказанные значения', fontsize=12, fontweight='bold')
                ax2.set_title('Точность прогноза: Факт vs Прогноз', fontsize=14, fontweight='bold', pad=15)
                ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
                ax2.grid(True, alpha=0.3)
                ax2.set_facecolor('#FAFAFA')
                
            else:
                # Если мало тестовых данных, показываем важность признаков (улучшенную)
                feature_importance = xgb_model.feature_importances_
                feature_names = ['Период', 'Месяц', 'Тренд', 'Лаг-1', 'Лаг-2', 'Лаг-3', 
                                'СМА-3', 'СМА-6', 'Сезон-sin', 'Сезон-cos']
                
                # Берём только топ-6 признаков для читаемости
                importance_data = list(zip(feature_names, feature_importance))
                importance_data.sort(key=lambda x: x[1], reverse=True)
                top_features = importance_data[:6]
                
                names, importances = zip(*top_features)
                
                # Цветовая схема
                colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
                
                bars = ax2.barh(names, importances, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
                
                # Добавляем проценты на столбцы
                for i, (bar, importance) in enumerate(zip(bars, importances)):
                    percentage = importance * 100
                    ax2.text(bar.get_width() + max(importances)*0.01, bar.get_y() + bar.get_height()/2, 
                            f'{percentage:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
                
                ax2.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
                ax2.set_title('Топ-6 важных признаков модели', fontsize=14, fontweight='bold', pad=15)
                ax2.grid(True, axis='x', alpha=0.3)
                ax2.set_facecolor('#FAFAFA')
                ax2.set_xlim(0, max(importances) * 1.15)
            
            # Общее улучшение дизайна
            plt.tight_layout(pad=3.0)
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохранение результатов
            self.forecast_results = {
                'dates': forecast_dates,
                'values': forecast_values,
                'model': 'XGBoost',
                'mae': mae,
                'r2': r2,
                'feature_importance': dict(zip(['Период', 'Месяц', 'Тренд', 'Лаг-1', 'Лаг-2', 'Лаг-3', 
                                                'СМА-3', 'СМА-6', 'Сезон-sin', 'Сезон-cos'], 
                                            xgb_model.feature_importances_))
            }
            
            self.update_status(f"XGBoost прогноз построен на {periods} месяцев (MAE: {mae:.1f}, R²: {r2:.3f})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении XGBoost прогноза: {str(e)}")
            import traceback
            print("Подробная ошибка XGBoost:")
            traceback.print_exc()

    def forecast_linear_regression(self):
        """Прогнозирование с использованием линейной регрессии (исправленная версия)"""
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Ошибка", "Библиотека scikit-learn не установлена!")
            return
            
        try:
            # Подготовка данных
            data = self.current_data.copy()
            region_filter = getattr(self, 'forecast_region_var', None)
            if region_filter and region_filter != 'Все' and 'Регион' in data.columns:
                data = data[data['Регион'] == region_filter]
            data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')
            data = data.dropna(subset=['Дата'])
            monthly_data = data.resample('MS', on='Дата')['Количество'].sum()

            if (monthly_data > 0).sum() < 6:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для линейной регрессии")
                return

            monthly_data = monthly_data[monthly_data > 0]
            
            # Подготовка признаков (отличается от SARIMA)
            X = np.arange(len(monthly_data)).reshape(-1, 1)
            y = monthly_data.values
            
            # Добавляем сезонные признаки
            months = np.array([date.month for date in monthly_data.index])
            month_sin = np.sin(2 * np.pi * months / 12).reshape(-1, 1)
            month_cos = np.cos(2 * np.pi * months / 12).reshape(-1, 1)
            
            # Объединяем все признаки
            X_extended = np.hstack([X, month_sin, month_cos, X**2])  # Тренд + сезонность + квадратичный тренд
            
            # Разделение на обучающую и тестовую выборки
            test_size = min(0.3, 4 / len(X_extended))
            X_train, X_test, y_train, y_test = train_test_split(X_extended, y, test_size=test_size, random_state=42)
            
            # Обучение модели
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Оценка качества
            y_pred_test = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            # Прогноз
            periods = self.forecast_period.get()
            X_future = np.arange(len(monthly_data), len(monthly_data) + periods).reshape(-1, 1)
            
            # Добавляем сезонные признаки для прогноза
            future_months = []
            last_date = monthly_data.index[-1]
            for i in range(periods):
                future_date = last_date + pd.DateOffset(months=i+1)
                future_months.append(future_date.month)
            
            future_months = np.array(future_months)
            future_month_sin = np.sin(2 * np.pi * future_months / 12).reshape(-1, 1)
            future_month_cos = np.cos(2 * np.pi * future_months / 12).reshape(-1, 1)
            
            X_future_extended = np.hstack([X_future, future_month_sin, future_month_cos, X_future**2])
            forecast_values = model.predict(X_future_extended)
            forecast_values = np.maximum(forecast_values, 0)  # Неотрицательные значения
            
            # Даты прогноза
            forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                        periods=periods, freq='M')
            
            # График
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # График 1: Прогноз
            ax1.plot(monthly_data.index, monthly_data.values, 
                label='Исторические данные', marker='o', linewidth=2, color='blue')
            
            # Аппроксимация на исторических данных
            y_fitted = model.predict(X_extended)
            ax1.plot(monthly_data.index, y_fitted, 
                label='Линейная аппроксимация', color='green', linestyle=':', linewidth=2, alpha=0.8)
            
            # Прогноз
            ax1.plot(forecast_dates, forecast_values, 
                label='Прогноз (Linear Regression)', color='red', marker='s', linestyle='--', linewidth=2)
            
            ax1.set_xlabel('Дата')
            ax1.set_ylabel('Количество случаев')
            ax1.set_title(f'Линейная регрессия: прогноз на {periods} месяцев\nMAE: {mae:.1f}, R²: {r2:.3f}', fontsize=14)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Остатки модели
            residuals = y - y_fitted
            ax2.scatter(range(len(residuals)), residuals, alpha=0.6, color='gray')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Индекс наблюдения')
            ax2.set_ylabel('Остатки')
            ax2.set_title('Анализ остатков модели', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Добавляем линию тренда остатков
            residual_trend = np.polyfit(range(len(residuals)), residuals, 1)
            ax2.plot(range(len(residuals)), np.poly1d(residual_trend)(range(len(residuals))), 
                    color='orange', linestyle='--', alpha=0.7, label='Тренд остатков')
            ax2.legend()
            
            plt.tight_layout()
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохранение результатов
            self.forecast_results = {
                'dates': forecast_dates,
                'values': forecast_values,
                'model': 'Linear Regression',
                'mae': mae,
                'r2': r2,
                'residuals_std': np.std(residuals)
            }
            
            self.update_status(f"Прогноз Linear Regression построен на {periods} месяцев (MAE: {mae:.1f}, R²: {r2:.3f})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении прогноза Linear Regression: {str(e)}")

    def forecast_ml(self):
        """Прогнозирование с использованием машинного обучения (ПОЛНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Ошибка", "Библиотека scikit-learn не установлена!")
            return
            
        try:
            # Подготовка данных
            data = self.current_data.copy()
            
            # ИСПРАВЛЕНИЕ: Более надежное преобразование дат
            try:
                data['Дата'] = pd.to_datetime(data['Дата'], errors='coerce')
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка обработки дат: {str(e)}")
                return
            
            # Удаляем строки с некорректными датами
            data = data.dropna(subset=['Дата'])
            
            if len(data) == 0:
                messagebox.showerror("Ошибка", "Нет корректных данных после обработки дат")
                return
            
            data['Месяц'] = data['Дата'].dt.month
            data['Год'] = data['Дата'].dt.year
            data['День_года'] = data['Дата'].dt.dayofyear
            
            # Агрегация по месяцам
            monthly_data = data.groupby(['Год', 'Месяц'])['Количество'].sum().reset_index()
            monthly_data['Период'] = monthly_data['Год'] * 12 + monthly_data['Месяц']
            
            if len(monthly_data) < 12:
                messagebox.showwarning("Предупреждение", "Недостаточно данных для ML прогнозирования (нужно минимум 12 месяцев)")
                return
            
            # Создание признаков
            monthly_data['Лаг_1'] = monthly_data['Количество'].shift(1)
            monthly_data['Лаг_2'] = monthly_data['Количество'].shift(2)
            monthly_data['Скользящее_среднее'] = monthly_data['Количество'].rolling(window=3, min_periods=1).mean()
            
            # Сезонные признаки
            monthly_data['Сезон_sin'] = np.sin(2 * np.pi * monthly_data['Месяц'] / 12)
            monthly_data['Сезон_cos'] = np.cos(2 * np.pi * monthly_data['Месяц'] / 12)
            
            # Убираем строки с NaN (только первые строки с лагами)
            monthly_data = monthly_data.dropna()
            
            if len(monthly_data) < 8:
                messagebox.showwarning("Предупреждение", "Недостаточно данных после обработки для ML модели")
                return
            
            # Подготовка признаков и целевой переменной
            feature_columns = ['Период', 'Месяц', 'Лаг_1', 'Лаг_2', 'Скользящее_среднее', 'Сезон_sin', 'Сезон_cos']
            X = monthly_data[feature_columns].values
            y = monthly_data['Количество'].values
            
            # Разделение на обучающую и тестовую выборки
            test_size = min(0.3, 3 / len(X))  # Минимум 3 наблюдения на тест
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
            
            # Обучение модели Random Forest
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Оценка качества модели
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Прогнозирование
            periods = self.forecast_period.get()
            forecast_values = []
            forecast_dates = []
            
            # Последние значения для построения прогноза
            last_values = monthly_data.tail(3)
            last_period = monthly_data['Период'].iloc[-1]
            last_year = int(monthly_data['Год'].iloc[-1])
            last_month = int(monthly_data['Месяц'].iloc[-1])
            
            for i in range(periods):
                # ИСПРАВЛЕНИЕ: Правильное вычисление новой даты
                new_period = last_period + i + 1
                
                # Правильный расчет года и месяца
                months_to_add = i + 1
                new_year = last_year
                new_month = last_month + months_to_add
                
                # Корректировка для перехода через годы
                while new_month > 12:
                    new_month -= 12
                    new_year += 1
                
                # Сезонные признаки
                season_sin = np.sin(2 * np.pi * new_month / 12)
                season_cos = np.cos(2 * np.pi * new_month / 12)
                
                # Лаговые признаки
                if i == 0:
                    lag_1 = monthly_data['Количество'].iloc[-1]
                    lag_2 = monthly_data['Количество'].iloc[-2] if len(monthly_data) > 1 else lag_1
                    moving_avg = monthly_data['Количество'].iloc[-3:].mean() if len(monthly_data) > 2 else lag_1
                elif i == 1:
                    lag_1 = forecast_values[0]
                    lag_2 = monthly_data['Количество'].iloc[-1]
                    moving_avg = np.mean([monthly_data['Количество'].iloc[-2], monthly_data['Количество'].iloc[-1], forecast_values[0]])
                else:
                    lag_1 = forecast_values[i-1]
                    lag_2 = forecast_values[i-2] if i > 1 else monthly_data['Количество'].iloc[-1]
                    if i >= 2:
                        moving_avg = np.mean(forecast_values[i-3:i])
                    else:
                        moving_avg = np.mean([monthly_data['Количество'].iloc[-1]] + forecast_values[:i])
                
                # Создание вектора признаков
                X_new = np.array([[new_period, new_month, lag_1, lag_2, moving_avg, season_sin, season_cos]])
                
                # Прогноз
                forecast_value = max(0, model.predict(X_new)[0])
                forecast_values.append(forecast_value)
                
                # ИСПРАВЛЕНИЕ: Безопасное создание дат прогноза
                try:
                    # Проверяем корректность года и месяца
                    if 1 <= new_month <= 12 and 1900 <= new_year <= 2100:
                        forecast_date = pd.Timestamp(year=new_year, month=new_month, day=1)
                    else:
                        # Fallback: используем последнюю дату + offset
                        last_date = pd.Timestamp(year=last_year, month=last_month, day=1)
                        forecast_date = last_date + pd.DateOffset(months=months_to_add)
                    
                    forecast_dates.append(forecast_date)
                    
                except (ValueError, OverflowError) as e:
                    # Дополнительный fallback
                    print(f"Ошибка создания даты для {new_year}-{new_month}: {e}")
                    last_date = pd.Timestamp(year=last_year, month=last_month, day=1)
                    forecast_date = last_date + pd.DateOffset(months=months_to_add)
                    forecast_dates.append(forecast_date)
            
            # Настройка matplotlib для корректного отображения
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # Создание графика
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # График 1: Прогноз
            # ИСПРАВЛЕНИЕ: Безопасное создание исторических дат
            try:
                historical_dates = []
                for _, row in monthly_data.iterrows():
                    try:
                        year = int(row['Год'])
                        month = int(row['Месяц'])
                        if 1 <= month <= 12 and 1900 <= year <= 2100:
                            date = pd.Timestamp(year=year, month=month, day=1)
                            historical_dates.append(date)
                        else:
                            # Пропускаем некорректные даты
                            continue
                    except (ValueError, TypeError):
                        continue
                
                if len(historical_dates) > 0:
                    # Берем соответствующие значения для корректных дат
                    valid_values = monthly_data['Количество'].iloc[:len(historical_dates)]
                    
                    ax1.plot(historical_dates, valid_values, 
                            label='📊 Исторические данные', marker='o', linewidth=2.5, 
                            color='#2E86AB', markersize=6, alpha=0.8)
                    
                    # Прогноз
                    if forecast_dates and forecast_values:
                        ax1.plot(forecast_dates, forecast_values, 
                                label='🚀 Прогноз Random Forest', color='#E74C3C', marker='s', 
                                linestyle='--', linewidth=3, markersize=7, alpha=0.9)
                        
                        # Добавляем вертикальную линию разделения
                        if historical_dates and forecast_dates:
                            ax1.axvline(x=historical_dates[-1], color='gray', linestyle=':', alpha=0.7, linewidth=2)
                            ax1.text(historical_dates[-1], ax1.get_ylim()[1]*0.9, ' Прогноз начинается здесь', 
                                    rotation=90, verticalalignment='top', fontsize=10, color='gray')
                else:
                    # Fallback к числовым индексам
                    ax1.plot(range(len(monthly_data)), monthly_data['Количество'], 
                            label='📊 Исторические данные', marker='o', linewidth=2.5, color='#2E86AB')
                    if forecast_values:
                        forecast_start = len(monthly_data)
                        ax1.plot(range(forecast_start, forecast_start + len(forecast_values)), forecast_values, 
                                label='🚀 Прогноз Random Forest', color='#E74C3C', marker='s', linestyle='--', linewidth=3)
            
            except Exception as e:
                print(f"Ошибка при создании графика дат: {e}")
                # Простой fallback
                ax1.plot(range(len(monthly_data)), monthly_data['Количество'], 
                        label='📊 Исторические данные', marker='o', linewidth=2.5, color='#2E86AB')
                if forecast_values:
                    forecast_start = len(monthly_data)
                    ax1.plot(range(forecast_start, forecast_start + len(forecast_values)), forecast_values, 
                            label='🚀 Прогноз Random Forest', color='#E74C3C', marker='s', linestyle='--', linewidth=3)
            
            # Улучшенное оформление первого графика
            ax1.set_xlabel('Период', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Количество случаев', fontsize=12, fontweight='bold')
            ax1.set_title(f'🎯 Random Forest: Прогноз заболеваемости на {periods} месяцев\n'
                        f'Точность: R² = {r2:.3f} | Ошибка: MAE = {mae:.1f}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=11)
            ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax1.set_facecolor('#FAFAFA')
            
            # График 2: Уникальный анализ для Random Forest (отличается от XGBoost)
            if len(X_test) > 0 and len(y_test) > 0:
                # Вариант A: Матрица ошибок и распределение остатков (УНИКАЛЬНЫЙ ДЛЯ RF)
                residuals = y_pred - y_test
                
                # Создаем subplot в subplot для более детального анализа
                from matplotlib.gridspec import GridSpec
                gs = GridSpec(2, 2, figure=fig, left=0.1, right=0.95, top=0.45, bottom=0.05, 
                            wspace=0.3, hspace=0.4)
                
                # Убираем старый ax2 и создаем новые
                ax2.remove()
                
                # Подграфик 1: Гистограмма остатков
                ax2_1 = fig.add_subplot(gs[0, 0])
                n, bins, patches = ax2_1.hist(residuals, bins=15, alpha=0.7, color='skyblue', 
                                            edgecolor='black', density=True)
                
                # Добавляем нормальное распределение для сравнения
                mu, sigma = np.mean(residuals), np.std(residuals)
                x_norm = np.linspace(residuals.min(), residuals.max(), 100)
                y_norm = ((1/(sigma * np.sqrt(2 * np.pi))) * 
                        np.exp(-0.5 * ((x_norm - mu) / sigma) ** 2))
                ax2_1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Норм. распр.')
                ax2_1.axvline(x=0, color='green', linestyle='--', alpha=0.8, linewidth=2)
                ax2_1.set_title('🔔 Распределение остатков', fontsize=12, fontweight='bold')
                ax2_1.set_xlabel('Остатки')
                ax2_1.set_ylabel('Плотность')
                ax2_1.legend()
                ax2_1.grid(True, alpha=0.3)
                
                # Подграфик 2: Q-Q plot для проверки нормальности
                ax2_2 = fig.add_subplot(gs[0, 1])
                from scipy import stats
                stats.probplot(residuals, dist="norm", plot=ax2_2)
                ax2_2.set_title('📈 Q-Q график нормальности', fontsize=12, fontweight='bold')
                ax2_2.grid(True, alpha=0.3)
                
                # Подграфик 3: Остатки vs предсказанные значения
                ax2_3 = fig.add_subplot(gs[1, 0])
                ax2_3.scatter(y_pred, residuals, alpha=0.6, color='coral', s=60, edgecolor='white')
                ax2_3.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
                
                # Добавляем LOWESS сглаживание для выявления паттернов
                try:
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(residuals, y_pred, frac=0.3)
                    ax2_3.plot(smoothed[:, 0], smoothed[:, 1], color='blue', linewidth=3, alpha=0.8)
                except:
                    # Если statsmodels недоступен, используем полиномиальное сглаживание
                    z = np.polyfit(y_pred, residuals, 2)
                    p = np.poly1d(z)
                    x_smooth = np.linspace(y_pred.min(), y_pred.max(), 100)
                    ax2_3.plot(x_smooth, p(x_smooth), color='blue', linewidth=3, alpha=0.8)
                
                ax2_3.set_title('🎯 Остатки vs Предсказания', fontsize=12, fontweight='bold')
                ax2_3.set_xlabel('Предсказанные значения')
                ax2_3.set_ylabel('Остатки')
                ax2_3.grid(True, alpha=0.3)
                
                # Подграфик 4: Круговая диаграмма качества предсказаний
                ax2_4 = fig.add_subplot(gs[1, 1])
                
                # Классификация качества предсказаний
                abs_errors = np.abs(residuals)
                error_threshold_low = np.percentile(abs_errors, 33)
                error_threshold_high = np.percentile(abs_errors, 67)
                
                excellent = np.sum(abs_errors <= error_threshold_low)
                good = np.sum((abs_errors > error_threshold_low) & (abs_errors <= error_threshold_high))
                poor = np.sum(abs_errors > error_threshold_high)
                
                sizes = [excellent, good, poor]
                labels = ['Отличные\n(≤33%)', 'Хорошие\n(33-67%)', 'Слабые\n(≥67%)']
                colors = ['#2ECC71', '#F39C12', '#E74C3C']
                explode = (0.05, 0.05, 0.1)
                
                wedges, texts, autotexts = ax2_4.pie(sizes, labels=labels, colors=colors, 
                                                    explode=explode, autopct='%1.1f%%', 
                                                    startangle=90, shadow=True)
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(9)
                
                ax2_4.set_title('🏆 Качество предсказаний', fontsize=12, fontweight='bold')
                
                # Общая статистика внизу
                overall_stats = f'📊 ОБЩАЯ СТАТИСТИКА МОДЕЛИ:\n' \
                            f'R² = {r2:.3f} | MAE = {mae:.1f} | RMSE = {np.sqrt(np.mean(residuals**2)):.1f}\n' \
                            f'Среднее остатков: {np.mean(residuals):.2f} | Медиана: {np.median(residuals):.2f}'
                
                fig.text(0.5, 0.02, overall_stats, ha='center', va='bottom', fontsize=11, 
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))

            elif len(monthly_data) >= 12:
                # Вариант B: Анализ важности деревьев Random Forest (УНИКАЛЬНЫЙ)
                ax2.clear()
                
                # Получаем важность признаков из индивидуальных деревьев
                n_trees_to_show = min(10, model.n_estimators)
                individual_importances = []
                
                for i in range(n_trees_to_show):
                    tree_importance = model.estimators_[i].feature_importances_
                    individual_importances.append(tree_importance)
                
                individual_importances = np.array(individual_importances)
                feature_names = ['Период', 'Месяц', 'Лаг 1', 'Лаг 2', 'Скольз. ср.', 'Сезон sin', 'Сезон cos']
                
                # Создаем violin plot для показа разброса важности по деревьям
                positions = range(len(feature_names))
                violin_data = [individual_importances[:, i] for i in range(len(feature_names))]
                
                violin_parts = ax2.violinplot(violin_data, positions=positions, 
                                            showmeans=True, showmedians=True, showextrema=True)
                
                # Красим violin plots в разные цвета
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
                for pc, color in zip(violin_parts['bodies'], colors):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                
                # Добавляем средние значения как точки
                means = [np.mean(data) for data in violin_data]
                ax2.scatter(positions, means, color='red', s=100, zorder=3, 
                        marker='D', edgecolor='white', linewidth=2, label='Среднее')
                
                # Добавляем значения на график
                for i, (pos, mean_val) in enumerate(zip(positions, means)):
                    ax2.text(pos, mean_val + max(means)*0.02, f'{mean_val:.3f}', 
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                ax2.set_xticks(positions)
                ax2.set_xticklabels(feature_names, rotation=45, ha='right')
                ax2.set_ylabel('Важность признака в деревьях', fontsize=12, fontweight='bold')
                ax2.set_title(f'🌲 Разброс важности признаков по {n_trees_to_show} деревьям RF', 
                            fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='y')
                ax2.legend()
                
                # Добавляем статистику разброса
                stability_scores = [np.std(data) for data in violin_data]
                most_stable = feature_names[np.argmin(stability_scores)]
                most_variable = feature_names[np.argmax(stability_scores)]
                
                stats_text = f'Стабильность признаков:\n' \
                            f'Самый стабильный: {most_stable}\n' \
                            f'Самый изменчивый: {most_variable}\n' \
                            f'Деревьев проанализировано: {n_trees_to_show}'
                
                ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
            else:
                # Вариант C: Дерево решений - визуализация структуры одного дерева (УНИКАЛЬНЫЙ)
                ax2.clear()
                
                # Выбираем одно из лучших деревьев для визуализации
                tree_idx = 0  # Можно выбрать случайное или лучшее дерево
                tree = model.estimators_[tree_idx]
                
                # Создаем упрощенную визуализацию дерева
                from matplotlib.patches import Rectangle, FancyBboxPatch
                
                # Получаем информацию о дереве
                n_nodes = tree.tree_.node_count
                children_left = tree.tree_.children_left
                children_right = tree.tree_.children_right
                feature = tree.tree_.feature
                threshold = tree.tree_.threshold
                
                # Ограничиваем глубину для читаемости
                max_depth_to_show = 3
                feature_names = ['Период', 'Месяц', 'Лаг 1', 'Лаг 2', 'Скольз. ср.', 'Сезон sin', 'Сезон cos']
                
                # Функция для рекурсивного рисования узлов
                def draw_tree_recursive(node_id, x, y, width, depth):
                    if depth > max_depth_to_show or node_id == -1:
                        return
                    
                    # Цвет узла в зависимости от глубины
                    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
                    color = colors[min(depth, len(colors)-1)]
                    
                    # Рисуем узел
                    if children_left[node_id] != children_right[node_id]:  # Не листовой узел
                        # Условие разбиения
                        if feature[node_id] < len(feature_names):
                            label = f'{feature_names[feature[node_id]]}\n≤ {threshold[node_id]:.2f}'
                        else:
                            label = f'Feature {feature[node_id]}\n≤ {threshold[node_id]:.2f}'
                        
                        box = FancyBboxPatch((x-width/2, y-0.05), width, 0.1, 
                                        boxstyle="round,pad=0.01", 
                                        facecolor=color, edgecolor='black', linewidth=1)
                        ax2.add_patch(box)
                        ax2.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold')
                        
                        # Рисуем линии к дочерним узлам
                        left_x = x - width/2
                        right_x = x + width/2
                        child_y = y - 0.2
                        
                        ax2.plot([x, left_x], [y-0.05, child_y+0.05], 'k-', linewidth=1)
                        ax2.plot([x, right_x], [y-0.05, child_y+0.05], 'k-', linewidth=1)
                        
                        # Рекурсивно рисуем дочерние узлы
                        draw_tree_recursive(children_left[node_id], left_x, child_y, width/2, depth+1)
                        draw_tree_recursive(children_right[node_id], right_x, child_y, width/2, depth+1)
                    else:
                        # Листовой узел
                        box = FancyBboxPatch((x-width/4, y-0.03), width/2, 0.06, 
                                        boxstyle="round,pad=0.01", 
                                        facecolor='lightpink', edgecolor='black', linewidth=1)
                        ax2.add_patch(box)
                        ax2.text(x, y, 'Лист', ha='center', va='center', fontsize=7)
                
                # Начинаем рисование с корня
                draw_tree_recursive(0, 0.5, 0.9, 0.8, 0)
                
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_aspect('equal')
                ax2.axis('off')
                ax2.set_title(f'🌳 Структура дерева #{tree_idx+1} (глубина ≤{max_depth_to_show})', 
                            fontsize=14, fontweight='bold', pad=20)
                
                # Добавляем легенду важности признаков
                feature_importance = model.feature_importances_
                importance_text = "🏆 Общая важность признаков:\n"
                sorted_features = sorted(zip(feature_names, feature_importance), 
                                    key=lambda x: x[1], reverse=True)
                
                for i, (fname, importance) in enumerate(sorted_features[:5]):
                    importance_text += f"{i+1}. {fname}: {importance:.3f}\n"
                
                ax2.text(0.02, 0.4, importance_text, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lavender', alpha=0.8))
                
                # Информация о модели
                model_info = f"📊 Информация о RF:\n" \
                            f"Всего деревьев: {model.n_estimators}\n" \
                            f"Макс. глубина: {model.max_depth}\n" \
                            f"Мин. образцов в листе: {model.min_samples_leaf}\n" \
                            f"Случайных признаков: {model.max_features}"
                
                ax2.text(0.98, 0.4, model_info, transform=ax2.transAxes, fontsize=10,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
            
            # Общие улучшения для второго графика
            try:
                ax2.set_facecolor('#FAFAFA')
                ax2.tick_params(axis='both', which='major', labelsize=10)
            except:
                pass  # Может быть удален в варианте A
            
            # Общее улучшение дизайна
            plt.tight_layout(pad=3.0)
            
            # Встраивание графика
            canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Сохранение результатов
            feature_names = ['Период', 'Месяц', 'Лаг 1', 'Лаг 2', 'Скольз. ср.', 'Сезон sin', 'Сезон cos']
            self.forecast_results = {
                'dates': forecast_dates,
                'values': forecast_values,
                'model': 'Random Forest',
                'mae': mae,
                'r2': r2,
                'feature_importance': dict(zip(feature_names, model.feature_importances_))
            }
            
            self.update_status(f"ML прогноз построен на {periods} месяцев (MAE: {mae:.1f}, R²: {r2:.3f})")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при построении ML прогноза: {str(e)}")
            # Показываем подробную ошибку для отладки
            import traceback
            print("Подробная ошибка Random Forest:")
            traceback.print_exc()

    def _plot_feature_importance_enhanced(self, ax, model):
        """Вспомогательный метод для отрисовки улучшенной важности признаков"""
        feature_importance = model.feature_importances_
        feature_names = ['Период', 'Месяц', 'Лаг 1', 'Лаг 2', 'Скольз. ср.', 'Сезон sin', 'Сезон cos']
        
        # Сортируем по важности
        importance_data = list(zip(feature_names, feature_importance))
        importance_data.sort(key=lambda x: x[1], reverse=True)
        sorted_names, sorted_importance = zip(*importance_data)
        
        # Создаем градиентную цветовую схему
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(sorted_names)))
        
        # Горизонтальная диаграмма с улучшенным дизайном
        bars = ax.barh(range(len(sorted_names)), sorted_importance, 
                    color=colors, alpha=0.8, edgecolor='white', linewidth=2)
        
        # Добавляем значения и проценты
        for i, (bar, importance) in enumerate(zip(bars, sorted_importance)):
            percentage = importance * 100
            ax.text(bar.get_width() + max(sorted_importance)*0.01, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{percentage:.1f}%', ha='left', va='center', 
                    fontsize=11, fontweight='bold', color='darkblue')
            
            # Добавляем ранг
            ax.text(-max(sorted_importance)*0.02, bar.get_y() + bar.get_height()/2, 
                    f'#{i+1}', ha='right', va='center', 
                    fontsize=10, fontweight='bold', color='darkred')
        
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=11)
        ax.set_xlabel('Важность признака', fontsize=12, fontweight='bold')
        ax.set_title('🏆 Рейтинг важности признаков (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.set_xlim(-max(sorted_importance)*0.05, max(sorted_importance) * 1.2)
        
        # Добавляем анализ важности
        top_3_sum = sum(sorted_importance[:3])
        stats_text = f'Анализ важности:\nТоп-3 признака: {top_3_sum*100:.1f}%\nДоминирующий: {sorted_names[0]}\nВсего признаков: {len(sorted_names)}'
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    def save_results(self):
        """Сохранение результатов анализа"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для сохранения!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("Excel файлы", "*.xlsx")],
            title="Сохранить результаты анализа"
        )
        
        if filename:
            try:
                # Выбираем данные для сохранения
                data_to_save = self.processed_data if self.processed_data is not None else self.current_data
                
                if filename.endswith('.csv'):
                    data_to_save.to_csv(filename, index=False, encoding='utf-8')
                else:
                    data_to_save.to_excel(filename, index=False)
                    
                messagebox.showinfo("Успех", f"Результаты сохранены: {os.path.basename(filename)}")
                self.update_status(f"Результаты сохранены в {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")

    def forecast_linear_regression(self):
            """Прогнозирование с использованием линейной регрессии"""
            if not SKLEARN_AVAILABLE:
                messagebox.showerror("Ошибка", "Библиотека scikit-learn не установлена!")
                return
                
            try:
                # Подготовка данных
                data = self.current_data.copy()
                region_filter = getattr(self, 'forecast_region_var', None)
                if region_filter and region_filter != 'Все' and 'Регион' in data.columns:
                    data = data[data['Регион'] == region_filter]
                data['Дата'] = pd.to_datetime(data['Дата'])
                monthly_data = data.groupby(pd.Grouper(key='Дата', freq='M'))['Количество'].sum()
                monthly_data = monthly_data[monthly_data > 0]
                
                if len(monthly_data) < 6:
                    messagebox.showwarning("Предупреждение", "Недостаточно данных для линейной регрессии")
                    return
                
                # Подготовка признаков
                X = np.arange(len(monthly_data)).reshape(-1, 1)
                y = monthly_data.values
                
                # Полиномиальные признаки
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X)
                
                # Обучение модели
                model = LinearRegression()
                model.fit(X_poly, y)
                
                # Прогноз
                periods = self.forecast_period.get()
                X_future = np.arange(len(monthly_data), len(monthly_data) + periods).reshape(-1, 1)
                X_future_poly = poly_features.transform(X_future)
                forecast_values = model.predict(X_future_poly)
                forecast_values = np.maximum(forecast_values, 0)
                
                # Даты прогноза
                last_date = monthly_data.index[-1]
                forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                            periods=periods, freq='M')
                
                # График
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Исторические данные
                ax.plot(monthly_data.index, monthly_data.values, 
                    label='Исторические данные', marker='o', linewidth=2, color='blue')
                
                # Аппроксимация на исторических данных
                y_fitted = model.predict(X_poly)
                ax.plot(monthly_data.index, y_fitted, 
                    label='Линейная аппроксимация', color='green', linestyle=':', linewidth=2)
                
                # Прогноз
                ax.plot(forecast_dates, forecast_values, 
                    label='Прогноз (Polynomial Regression)', color='red', marker='s', linestyle='--', linewidth=2)
                
                ax.set_xlabel('Дата')
                ax.set_ylabel('Количество случаев')
                ax.set_title(f'Прогноз заболеваемости на {periods} месяцев (Polynomial Regression)', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Встраивание графика
                canvas = FigureCanvasTkAgg(fig, master=self.forecast_plot_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                # Сохранение результатов
                r2 = model.score(X_poly, y)
                self.forecast_results = {
                    'dates': forecast_dates,
                    'values': forecast_values,
                    'model': 'Linear Regression',
                    'r2': r2
                }
                
                self.update_status(f"Прогноз Linear Regression построен на {periods} месяцев (R²={r2:.3f})")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при построении прогноза Linear Regression: {str(e)}")
                                        
    def create_report(self):
        """Создание отчета"""
        if self.current_data is None:
            messagebox.showwarning("Предупреждение", "Нет данных для создания отчета!")
            return
            
        report_type = self.report_type.get()
        
        # Разблокируем текстовое поле
        self.report_text.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        
        try:
            # Генерация отчета в зависимости от типа
            if report_type == "summary":
                self.generate_summary_report()
            elif report_type == "detailed":
                self.generate_detailed_report()
            elif report_type == "forecast":
                self.generate_forecast_report()
            elif report_type == "comparative":
                self.generate_comparative_report()
                
        except Exception as e:
            error_text = f"Ошибка при создании отчета: {str(e)}"
            self.report_text.insert(1.0, error_text)
        
        finally:
            # Блокируем текстовое поле
            self.report_text.config(state=tk.DISABLED)

    def generate_summary_report(self):
        """Генерация сводного отчета"""
        try:
            report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         СРАВНИТЕЛЬНЫЙ ОТЧЕТ                                  ║
║                       ЗАБОЛЕВАЕМОСТЬ НАСЕЛЕНИЯ РК                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}

1. СРАВНЕНИЕ ПО РЕГИОНАМ - Топ-10 регионов по заболеваемости
"""
            
            # Рейтинг регионов
            regional_totals = self.current_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
            
            for i, (region, total) in enumerate(regional_totals.head(10).items(), 1):
                percentage = (total / self.current_data['Количество'].sum()) * 100
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                report += f"\n{medal} {i:2d}. {region:<20} {total:>8,.0f} ({percentage:5.1f}%)"

            self.report_text.insert(1.0, report)
            self.update_status("Сравнительный отчет создан успешно")
            
        except Exception as e:
                    error_text = f"Ошибка при создании сводного отчета: {str(e)}"
                    self.report_text.insert(1.0, error_text)

    def generate_summary_report(self):
        """Генерация сводного отчета"""
        try:
            report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  СВОДНЫЙ ОТЧЕТ ПО ЗАБОЛЕВАЕМОСТИ НАСЕЛЕНИЯ                   ║
║                              РЕСПУБЛИКА КАЗАХСТАН                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}
📊 Версия системы: 1.2

══════════════════════════════════════════════════════════════════════════════
                              1. ОБЩАЯ ИНФОРМАЦИЯ
══════════════════════════════════════════════════════════════════════════════

🗓️  Период анализа: {self.current_data['Дата'].min()} — {self.current_data['Дата'].max()}
📈  Общее количество случаев: {self.current_data['Количество'].sum():,}
🏥  Количество записей: {len(self.current_data):,}
🌍  Количество регионов: {self.current_data['Регион'].nunique()}
💊  Типов заболеваний: {self.current_data['Заболевание'].nunique()}

══════════════════════════════════════════════════════════════════════════════
                         2. СТАТИСТИКА ПО ЗАБОЛЕВАНИЯМ
══════════════════════════════════════════════════════════════════════════════
"""
            # Статистика по заболеваниям
            disease_stats = self.current_data.groupby('Заболевание')['Количество'].agg(['sum', 'mean', 'std']).round(1)
            
            for disease, stats in disease_stats.iterrows():
                report += f"""
📍 {disease}:
   • Всего случаев: {stats['sum']:,.0f}
   • Среднее в месяц: {stats['mean']:,.1f}
   • Стандартное отклонение: {stats['std']:,.1f}"""
            
            report += f"""

══════════════════════════════════════════════════════════════════════════════
                         3. ТОП-5 РЕГИОНОВ ПО ЗАБОЛЕВАЕМОСТИ
══════════════════════════════════════════════════════════════════════════════
"""
            top_regions = self.current_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False).head(5)
            for i, (region, count) in enumerate(top_regions.items(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🏅"
                percentage = (count / self.current_data['Количество'].sum()) * 100
                report += f"\n{medal} {i}. {region}: {count:,} случаев ({percentage:.1f}%)"
            
            # Возрастное распределение (если есть данные)
            if 'Возраст' in self.current_data.columns:
                report += f"""

══════════════════════════════════════════════════════════════════════════════
                            4. ВОЗРАСТНОЕ РАСПРЕДЕЛЕНИЕ
══════════════════════════════════════════════════════════════════════════════
"""
                age_bins = pd.cut(self.current_data['Возраст'], bins=[0, 14, 30, 45, 60, 100], 
                                 labels=['0-14 лет', '15-30 лет', '31-45 лет', '46-60 лет', '60+ лет'])
                age_dist = self.current_data.groupby(age_bins)['Количество'].sum()
                
                for age_group, count in age_dist.items():
                    percentage = (count / self.current_data['Количество'].sum()) * 100
                    report += f"\n👥 {age_group}: {count:,} случаев ({percentage:.1f}%)"
            
            # Сезонный анализ
            report += f"""

══════════════════════════════════════════════════════════════════════════════
                               5. СЕЗОННЫЙ АНАЛИЗ
══════════════════════════════════════════════════════════════════════════════
"""
            self.current_data['Дата_dt'] = pd.to_datetime(self.current_data['Дата'])
            seasonal_data = self.current_data.groupby(self.current_data['Дата_dt'].dt.quarter)['Количество'].sum()
            seasons = {1: 'I квартал (зима-весна)', 2: 'II квартал (весна-лето)', 
                      3: 'III квартал (лето-осень)', 4: 'IV квартал (осень-зима)'}
            
            for quarter, count in seasonal_data.items():
                percentage = (count / self.current_data['Количество'].sum()) * 100
                report += f"\n🗓️  {seasons[quarter]}: {count:,} случаев ({percentage:.1f}%)"
            
            report += f"""

══════════════════════════════════════════════════════════════════════════════
                                6. РЕКОМЕНДАЦИИ
══════════════════════════════════════════════════════════════════════════════

🔍 Основные выводы:
• Наибольшая заболеваемость зафиксирована в регионе: {top_regions.index[0]}
• Доминирующее заболевание: {disease_stats.sort_values('sum', ascending=False).index[0]}
• Пиковый сезон: {seasons[seasonal_data.idxmax()]}

💡 Рекомендации:
• Усилить профилактические мероприятия в регионах с высокой заболеваемостью
• Подготовиться к сезонному росту заболеваемости
• Провести дополнительный анализ факторов риска

══════════════════════════════════════════════════════════════════════════════
Отчет сформирован автоматически системой анализа заболеваний населения РК v1.2
© 2025 Министерство здравоохранения Республики Казахстан
══════════════════════════════════════════════════════════════════════════════
            """
            
            self.report_text.insert(1.0, report)
            self.update_status("Сводный отчет создан успешно")
            
        except Exception as e:
            error_text = f"Ошибка при создании сводного отчета: {str(e)}"
            self.report_text.insert(1.0, error_text)

    def generate_detailed_report(self):
        """Генерация детального отчета"""
        try:
            report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  ДЕТАЛЬНЫЙ ОТЧЕТ ПО ЗАБОЛЕВАЕМОСТИ НАСЕЛЕНИЯ                 ║
║                              РЕСПУБЛИКА КАЗАХСТАН                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}

══════════════════════════════════════════════════════════════════════════════
                               1. ПОМЕСЯЧНАЯ ДИНАМИКА
══════════════════════════════════════════════════════════════════════════════
"""
            # Помесячная статистика
            self.current_data['Дата_dt'] = pd.to_datetime(self.current_data['Дата'])
            monthly_stats = self.current_data.groupby(self.current_data['Дата_dt'].dt.to_period('M'))['Количество'].sum()
            
            for period, count in monthly_stats.items():
                report += f"\n📊 {period}: {count:,} случаев"
                
            report += f"""

══════════════════════════════════════════════════════════════════════════════
                              2. РЕГИОНАЛЬНЫЙ АНАЛИЗ
══════════════════════════════════════════════════════════════════════════════
"""
            # Детальная статистика по регионам
            for region in sorted(self.current_data['Регион'].unique()):
                region_data = self.current_data[self.current_data['Регион'] == region]
                total_cases = region_data['Количество'].sum()
                
                report += f"""

🏥 {region}:
   📈 Всего случаев: {total_cases:,}
   📊 Записей: {len(region_data):,}
   🗓️  Период: {region_data['Дата'].min()} — {region_data['Дата'].max()}"""
                
                # Топ заболевания в регионе
                top_diseases = region_data.groupby('Заболевание')['Количество'].sum().sort_values(ascending=False).head(3)
                report += "\n   💊 Основные заболевания:"
                for j, (disease, count) in enumerate(top_diseases.items(), 1):
                    percentage = (count / total_cases) * 100
                    report += f"\n      {j}. {disease}: {count:,} ({percentage:.1f}%)"
            
            self.report_text.insert(1.0, report)
            self.update_status("Детальный отчет создан успешно")
            
        except Exception as e:
            error_text = f"Ошибка при создании детального отчета: {str(e)}"
            self.report_text.insert(1.0, error_text)

    def generate_forecast_report(self):
        """Генерация прогнозного отчета"""
        try:
            if self.forecast_results is None:
                report = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ПРОГНОЗНЫЙ ОТЧЕТ                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

⚠️  ВНИМАНИЕ: Прогноз еще не построен!

Для создания прогнозного отчета:
1. Перейдите на вкладку "Прогнозы"
2. Выберите модель прогнозирования
3. Настройте параметры
4. Нажмите "Построить прогноз"
5. Вернитесь к созданию отчета
                """
            else:
                model_name = self.forecast_results.get('model', 'Неизвестная')
                dates = self.forecast_results['dates']
                values = self.forecast_results['values']
                
                report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ПРОГНОЗНЫЙ ОТЧЕТ                                    ║
║                       ЗАБОЛЕВАЕМОСТЬ НАСЕЛЕНИЯ РК                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

📅 Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}
🔮 Модель прогнозирования: {model_name}

══════════════════════════════════════════════════════════════════════════════
                              1. ПАРАМЕТРЫ ПРОГНОЗА
══════════════════════════════════════════════════════════════════════════════

🎯 Модель: {model_name}
📊 Период прогноза: {len(values)} месяцев
🗓️  Прогнозный период: {dates[0].strftime('%Y-%m')} — {dates[-1].strftime('%Y-%m')}"""

                # Добавляем метрики качества если есть
                if 'r2' in self.forecast_results:
                    r2 = self.forecast_results['r2']
                    report += f"\n📈 Коэффициент детерминации (R²): {r2:.3f}"
                    
                if 'mae' in self.forecast_results:
                    mae = self.forecast_results['mae']
                    report += f"\n📉 Средняя абсолютная ошибка: {mae:.0f}"

                report += f"""

══════════════════════════════════════════════════════════════════════════════
                              2. ПРОГНОЗНЫЕ ЗНАЧЕНИЯ
══════════════════════════════════════════════════════════════════════════════
"""
                
                # Прогнозные значения по месяцам
                for date, value in zip(dates, values):
                    report += f"\n📅 {date.strftime('%Y-%m')}: {value:,.0f} случаев"
            
            self.report_text.insert(1.0, report)
            self.update_status("Прогнозный отчет создан успешно")
            
        except Exception as e:
            error_text = f"Ошибка при создании прогнозного отчета: {str(e)}"
            self.report_text.insert(1.0, error_text)

    def generate_comparative_report(self):
            """Генерация сравнительного отчета"""
            try:
                report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         СРАВНИТЕЛЬНЫЙ ОТЧЕТ                                  ║
║                       ЗАБОЛЕВАЕМОСТЬ НАСЕЛЕНИЯ РК                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

    📅 Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}

                1. СРАВНЕНИЕ ПО РЕГИОНАМ - Топ-10 регионов по заболеваемости
                """
                
                # Рейтинг регионов
                regional_totals = self.current_data.groupby('Регион')['Количество'].sum().sort_values(ascending=False)
                
                for i, (region, total) in enumerate(regional_totals.head(10).items(), 1):
                    percentage = (total / self.current_data['Количество'].sum()) * 100
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "📊"
                    report += f"\n{medal} {i:2d}. {region:<20} {total:>8,.0f} ({percentage:5.1f}%)"

                self.report_text.insert(1.0, report)
                self.update_status("Сравнительный отчет создан успешно")
                
            except Exception as e:
                error_text = f"Ошибка при создании сравнительного отчета: {str(e)}"
                self.report_text.insert(1.0, error_text)

    def export_report(self):
        """Экспорт отчета"""
        content = self.report_text.get(1.0, tk.END).strip()
        if not content:
            messagebox.showwarning("Предупреждение", "Нет отчета для экспорта!")
            return
            
        format_type = self.export_format.get()
        
        try:
            if format_type == "PDF":
                self.export_to_pdf(content)
            elif format_type == "Excel":
                self.export_to_excel(content)
            elif format_type == "Word":
                self.export_to_word(content)
            elif format_type == "HTML":
                self.export_to_html(content)
            elif format_type == "TXT":
                self.export_to_txt(content)
            else:
                messagebox.showwarning("Предупреждение", f"Неподдерживаемый формат: {format_type}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при экспорте: {str(e)}")

    def export_to_html(self, content):
        """Экспорт отчета в HTML"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML файлы", "*.html")],
            title="Сохранить отчет как HTML"
        )
        
        if not filename:
            return
            
        try:
            html_content = f"""
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Медицинский отчет - Система анализа заболеваний РК</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 24px;
        }}
        .header .subtitle {{
            color: #7f8c8d;
            margin-top: 10px;
        }}
        pre {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.4;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            font-size: 12px;
            color: #7f8c8d;
        }}
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Отчет системы анализа заболеваний</h1>
            <div class="subtitle">Республика Казахстан • Министерство здравоохранения</div>
            <div class="subtitle">Создано: {datetime.now().strftime('%d.%m.%Y в %H:%M')}</div>
        </div>
        
        <pre>{content}</pre>
        
        <div class="footer">
            <p><strong>Система анализа заболеваний населения РК v1.2</strong></p>
            <p>© 2025 Министерство здравоохранения Республики Казахстан</p>
            <p><em>Данный отчет сформирован автоматически и требует профессиональной интерпретации</em></p>
        </div>
    </div>
</body>
</html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            messagebox.showinfo("Успех", f"HTML отчет сохранен: {os.path.basename(filename)}")
            
            # Предлагаем открыть файл
            response = messagebox.askyesno("Открыть файл", "Открыть созданный HTML файл в браузере?")
            if response:
                webbrowser.open('file://' + os.path.abspath(filename))
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании HTML файла: {str(e)}")

    def export_to_pdf(self, content):
        """Экспорт отчета в PDF"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF файлы", "*.pdf")],
            title="Сохранить отчет как PDF"
        )
        
        if not filename:
            return
            
        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.add_font('Arial', '', 'C:\\Windows\\Fonts\\arial.ttf', uni=True)
            pdf.set_font('Arial', size=12)
            for line in content.splitlines():
                pdf.multi_cell(0, 10, txt=line)
            pdf.output(filename)
            messagebox.showinfo("Успех", f"PDF отчет сохранен: {os.path.basename(filename)}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании PDF: {str(e)}")

    def export_to_excel(self, content):
        """Экспорт отчета в Excel"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel файлы", "*.xlsx")],
            title="Сохранить отчет как Excel"
        )
        
        if not filename:
            return
            
        try:
            if OPENPYXL_AVAILABLE:
                # Используем openpyxl если доступен
                workbook = openpyxl.Workbook()
                worksheet = workbook.active
                worksheet.title = "Медицинский отчет"
                
                # Заголовок
                worksheet['A1'] = "Отчет системы анализа заболеваний населения РК"
                worksheet['A1'].font = Font(bold=True, size=16)
                worksheet['A1'].alignment = Alignment(horizontal='center')
                
                # Дата создания
                worksheet['A2'] = f"Дата создания: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                worksheet['A2'].font = Font(size=12)
                
                # Содержание отчета
                lines = content.split('\n')
                row = 4
                
                for line in lines:
                    if line.strip():
                        worksheet[f'A{row}'] = line.replace('═', '=').replace('║', '|')  # Заменяем спецсимволы
                        # Выделяем заголовки
                        if any(marker in line for marker in ['ОТЧЕТ', 'АНАЛИЗ', 'СТАТИСТИКА', '═══']):
                            worksheet[f'A{row}'].font = Font(bold=True)
                    row += 1
                
                # Автоматическая ширина колонок
                worksheet.column_dimensions['A'].width = 120
                
                workbook.save(filename)
                messagebox.showinfo("Успех", f"Отчет сохранен в Excel: {os.path.basename(filename)}")
                
            else:
                # Используем pandas
                lines = content.split('\n')
                df = pd.DataFrame({'Отчет': lines})
                df.to_excel(filename, index=False)
                messagebox.showinfo("Успех", f"Отчет сохранен в Excel: {os.path.basename(filename)}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании Excel файла: {str(e)}")

    def export_to_word(self, content):
        """Экспорт отчета в Word (RTF формат)"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".rtf",
            filetypes=[("RTF файлы", "*.rtf"), ("Word файлы", "*.docx")],
            title="Сохранить отчет как Word"
        )
        
        if not filename:
            return
            
        try:
            # Создаем RTF файл (читается Word'ом)
            rtf_header = r"""{\rtf1\ansi\deff0 {\fonttbl {\f0 Courier New;}}{\colortbl;\red0\green0\blue0;\red0\green0\blue255;}
\f0\fs20 """
            
            # Очищаем содержимое от спецсимволов RTF
            clean_content = content.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')
            clean_content = clean_content.replace('═', '=').replace('║', '|')
            
            rtf_content = rtf_header + clean_content.replace('\n', r'\par ') + r'}'
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(rtf_content)
                
            messagebox.showinfo("Успех", f"Отчет сохранен как RTF: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при создании RTF файла: {str(e)}")

    def export_to_txt(self, content):
        """Экспорт отчета в текстовый файл"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Текстовые файлы", "*.txt")],
            title="Сохранить отчет как текст"
        )
        
        if not filename:
            return
            
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            messagebox.showinfo("Успех", f"Отчет сохранен как текст: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при сохранении текстового файла: {str(e)}")


def generate_test_data():
    """Генерация тестовых данных для демонстрации"""
    try:
        np.random.seed(42)
        
        # Параметры генерации
        start_date = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
        regions = ['Алматы', 'Астана', 'Караганда', 'Шымкент', 'Актобе', 
                'Павлодар', 'Тараз', 'Усть-Каменогорск', 'Костанай', 'Атырау',
                'Петропавловск', 'Актау', 'Кокшетау', 'Семей', 'Талдыкорган']
        diseases = ['ОРВИ', 'Грипп', 'Пневмония', 'Диабет', 'Гипертония', 
                'Бронхит', 'Астма', 'Гастрит', 'Артрит', 'Мигрень']
        
        data = []
        record_id = 1
        
        print("Генерация тестовых данных...")
        
        for i, date in enumerate(start_date):
            if i % 365 == 0:  # Обновляем прогресс каждый год
                print(f"Обработка {date.year} года...")
                
            # Сезонный фактор
            month = date.month
            seasonal_factor = 1.8 if month in [11, 12, 1, 2, 3] else 0.6  # Зима vs лето
            
            # Генерация записей для каждого дня
            num_records = np.random.poisson(30)  # Среднее количество записей в день
            
            for _ in range(num_records):
                region = np.random.choice(regions, p=np.random.dirichlet(np.ones(len(regions))))
                disease = np.random.choice(diseases)
                
                # Возраст и количество в зависимости от заболевания
                if disease in ['ОРВИ', 'Грипп']:
                    age = np.random.normal(25, 15)
                    base_count = seasonal_factor * np.random.gamma(2, 3)
                elif disease == 'Пневмония':
                    age = np.random.normal(45, 20)
                    base_count = seasonal_factor * 0.7 * np.random.gamma(2, 2)
                elif disease == 'Диабет':
                    age = np.random.normal(55, 12)
                    base_count = np.random.gamma(1.5, 2)
                elif disease == 'Гипертония':
                    age = np.random.normal(60, 10)
                    base_count = np.random.gamma(1.8, 2)
                elif disease in ['Бронхит', 'Астма']:
                    age = np.random.normal(35, 18)
                    base_count = seasonal_factor * 0.8 * np.random.gamma(1.5, 2)
                else:
                    age = np.random.normal(40, 20)
                    base_count = np.random.gamma(1.2, 2)
                
                age = max(1, min(95, int(age)))  # Ограничение возраста
                count = max(1, int(base_count))
                
                # Региональные особенности
                if region in ['Алматы', 'Астана']:
                    count = int(count * 1.3)  # Больше случаев в крупных городах
                elif region in ['Атырау', 'Актау']:
                    count = int(count * 0.8)  # Меньше в отдаленных регионах
                
                data.append({
                    'ID': record_id,
                    'Дата': date,
                    'Регион': region,
                    'Заболевание': disease,
                    'Возраст': age,
                    'Пол': np.random.choice(['М', 'Ж'], p=[0.48, 0.52]),  # Слегка больше женщин
                    'Количество': count
                })
                
                record_id += 1
        
        df = pd.DataFrame(data)
        print(f"Сгенерировано {len(df)} записей")
        return df
        
    except Exception as e:
        print(f"Ошибка при генерации данных: {e}")
        # Возвращаем минимальный набор данных
        simple_data = {
            'ID': range(1, 101),
            'Дата': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Регион': np.random.choice(['Алматы', 'Астана', 'Караганда'], 100),
            'Заболевание': np.random.choice(['ОРВИ', 'Грипп', 'Пневмония'], 100),
            'Возраст': np.random.randint(1, 80, 100),
            'Пол': np.random.choice(['М', 'Ж'], 100),
            'Количество': np.random.randint(1, 20, 100)
        }
        return pd.DataFrame(simple_data)

def main():
    """Главная функция запуска приложения"""
    try:
        root = tk.Tk()
        
        # Установка иконки (если есть)
        try:
            root.iconbitmap('medical_icon.ico')
        except:
            pass  # Игнорируем отсутствие иконки
        
        # Создание приложения
        app = MedicalAnalysisSystem(root)
        
        # Предлагаем загрузить тестовые данные
        response = messagebox.askyesno("Система анализа заболеваний РК v1.2", 
                                    "Добро пожаловать в систему анализа заболеваний!\n\n"
                                    "Загрузить демонстрационные данные для ознакомления с системой?")
        if response:
            try:
                app.update_status("Генерация демонстрационных данных...")
                app.root.update()
                
                app.current_data = generate_test_data()
                app.update_data_display()
                app.update_map_filters()  # Обновляем фильтры карты после загрузки
                app.update_status(f"Загружены демонстрационные данные: {len(app.current_data)} записей")
                
                messagebox.showinfo("Данные загружены", 
                                f"Демонстрационные данные успешно загружены!\n\n"
                                f"📊 Записей: {len(app.current_data):,}\n"
                                f"🏥 Регионов: {app.current_data['Регион'].nunique()}\n"
                                f"💊 Заболеваний: {app.current_data['Заболевание'].nunique()}\n"
                                f"📅 Период: {app.current_data['Дата'].min()} - {app.current_data['Дата'].max()}\n\n"
                                f"Теперь вы можете исследовать функции системы!")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при генерации данных: {str(e)}")
                app.update_status("Ошибка загрузки демонстрационных данных")
        
        # Запуск главного цикла
        root.mainloop()
        
    except Exception as e:
        print(f"Критическая ошибка при запуске: {e}")
        messagebox.showerror("Критическая ошибка", 
                        f"Ошибка при запуске приложения:\n{str(e)}\n\n"
                        f"Проверьте установку необходимых библиотек:\n"
                        f"pip install pandas numpy matplotlib seaborn tkinter")

if __name__ == "__main__":
    main()