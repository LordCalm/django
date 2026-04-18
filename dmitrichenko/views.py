import os
import io
import csv
import base64
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.offline as pyo
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.cache import cache
from .forms import UserForm
import pyet
import spei as si

# Вспомогательная функция для конвертации графика в Base64
def get_graph():
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png).decode('utf-8')
    buffer.close()
    plt.close() # Очищаем память
    return graph

# Функция для главной страницы
def index(request):
    return render(request, 'dmitrichenko/index.html')

def about(request):
    data = {
        'title': 'О студенте',
        'dict': {'name': 'Дмитриченко', 'Дмитрий': 'Александрович', 'group': 'АТМ-25'},
    }
    return render(request, 'dmitrichenko/about.html', context=data)

# Функция для работы с CSV данными
def data_view(request):
    # Путь к папке с данными
    data_dir = os.path.join(settings.BASE_DIR, 'dmitrichenko', 'static', 'dmitrichenko', 'data')
    
    # Получаем список всех .csv файлов в папке
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    # Получаем имя выбранного файла из GET-запроса (по умолчанию первый в списке)
    selected_file = request.GET.get('file', files[0] if files else None)
    
    data_list = []
    headers = []  # Список для названий столбцов
    
    if selected_file:
        file_path = os.path.join(data_dir, selected_file)
        with open(file_path, newline='', encoding='cp1251') as f:
            reader = csv.DictReader(f, delimiter=';')
            data_list = list(reader)
            headers = reader.fieldnames  # Получаем имена столбцов

    context = {
        'files': files,
        'selected_file': selected_file,
        'data': data_list,
        'headers': headers,
    }
    return render(request, 'dmitrichenko/data.html', context)

def user_form_view(request):
    if request.method == "POST":
        form = UserForm(request.POST)
        if form.is_valid():
            # Собираем данные для отображения на другой странице
            context = {
                'data': form.cleaned_data
            }
            return render(request, 'dmitrichenko/user_result.html', context)
    else:
        form = UserForm()
    
    return render(request, 'dmitrichenko/user_form.html', {'form': form})

def get_arima_forecast(temp_ts, precip_ts, selected_file):
    """
    Быстрый расчет SARIMA с фиксированными параметрами для погодных данных.
    """
    cache_key = f"forecast_{selected_file}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    # Фиксированные параметры: 
    # order=(1,0,1) - базовая авторегрессия и скользящее среднее
    # seasonal_order=(0,1,1,12) - сезонная разность (12 месяцев) и сглаживание
    
    # Модель для температуры
    model_t = SARIMAX(temp_ts, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))
    fit_t = model_t.fit(disp=False)
    forecast_t = fit_t.forecast(steps=24)
    
    # Модель для осадков (осадки более шумные, но эти параметры тоже подойдут как базовые)
    model_p = SARIMAX(precip_ts, order=(1, 0, 1), seasonal_order=(0, 1, 1, 12))
    fit_p = model_p.fit(disp=False)
    forecast_p = fit_p.forecast(steps=24)
    
    cache.set(cache_key, (forecast_t, forecast_p), timeout=3600)
    
    return forecast_t, forecast_p

def load_climate_timeseries(selected_file):
    """
    Загружает CSV и возвращает:
    temp_ts  - помесячный ряд температуры
    precip_ts - помесячный ряд осадков
    files - список файлов
    """
    
    data_dir = os.path.join(settings.BASE_DIR, 'dmitrichenko', 'static', 'dmitrichenko', 'data')
    
    file_path = os.path.join(data_dir, selected_file)
    df = pd.read_csv(file_path, sep=';', encoding='cp1251')
    lat = pd.to_numeric(df['Широта'], errors='coerce').dropna().iloc[0]
    
    # Заменяем заглушки на NaN
    df = df.replace([999.9, -999.0, -999.9, '999.9', '-999.0', '-999.9'], np.nan)
    df = df.dropna(subset=['год'])

    # --- 1. ПРЕОБРАЗОВАНИЕ В ПОМЕСЯЧНЫЙ РЯД ---
    months = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    precip_cols = [m + ' осадки' for m in months]

    # === ПРЕОБРАЗОВАНИЕ В ЧИСЛОВОЙ ФОРМАТ ДО MELT ===
    df['год'] = pd.to_numeric(df['год'], errors='coerce')

    for col in months:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in precip_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_temp = pd.melt(df, id_vars=['год'], value_vars=months, var_name='месяц', value_name='temp')
    df_precip = pd.melt(df, id_vars=['год'], value_vars=precip_cols, var_name='месяц', value_name='precip')
    df_precip['месяц'] = df_precip['месяц'].str.replace(' осадки', '')

    df_long = pd.merge(df_temp, df_precip, on=['год', 'месяц'])
    month_map = {m: i+1 for i, m in enumerate(months)}
    df_long['month_num'] = df_long['месяц'].map(month_map)

    df_long['date'] = pd.to_datetime(
        df_long['год'].astype(int).astype(str) + '-' + 
        df_long['month_num'].astype(str) + '-01'
    )

    df_long = df_long.sort_values('date')
    df_long = df_long.set_index('date')

    # Оставляем только нужные колонки и гарантируем числовой формат
    df_long['temp'] = pd.to_numeric(df_long['temp'], errors='coerce')
    df_long['precip'] = pd.to_numeric(df_long['precip'], errors='coerce')

    # 1. Приводим к жесткому помесячному календарю.
    # asfreq('MS') сам создаст пустые строки (NaN) для пропущенных месяцев и лет.
    ts_data = df_long[['temp', 'precip']].asfreq('MS')

    # 2. Локальная интерполяция: соединяем короткие дырки (1-2 месяца) линией
    ts_data = ts_data.interpolate(method='linear', limit=2)

    # 3. Глобальное заполнение: если дырка большая (год и более), 
    # заполняем медианой именно для этого месяца за все доступные годы.
    ts_data['temp'] = ts_data['temp'].fillna(
        ts_data.groupby(ts_data.index.month)['temp'].transform('median')
    )
    # --- Более реалистичное заполнение осадков ---

    # Сохраняем маску пропусков
    mask_precip = ts_data['precip'].isna()

    # Статистика по каждому месяцу
    month_median = ts_data.groupby(ts_data.index.month)['precip'].transform('median')
    month_std = ts_data.groupby(ts_data.index.month)['precip'].transform('std')

    # Если std вдруг NaN или 0
    month_std = month_std.fillna(month_std.median())
    month_std = month_std.replace(0, month_std.median())

    # Генерируем значения:
    # медиана месяца + случайное отклонение
    np.random.seed(42)   # чтобы результат повторялся

    filled_values = (
        month_median[mask_precip] +
        np.random.normal(
            loc=0,
            scale=month_std[mask_precip] * 0.6
        )
    )

    # Осадки не могут быть отрицательными
    filled_values = np.clip(filled_values, 0, None)

    # Записываем
    ts_data.loc[mask_precip, 'precip'] = filled_values


    # 4. Финальная страховка на случай, если какой-то месяц отсутствует за ВСЕ годы (крайне редко)
    ts_data = ts_data.fillna(ts_data.median())

    df_long['temp'] = ts_data['temp']
    df_long['precip'] = ts_data['precip']

    # Извлекаем готовые Series
    temp_ts = ts_data['temp']
    precip_ts = ts_data['precip']
    return temp_ts, precip_ts, df_long, lat

def calculate_gtk(group):
    active = group['temp'] > 10
    
    active_temps = group.loc[active, 'temp'].sum()
    active_precip = group.loc[active, 'precip'].sum()

    if active_temps > 0:
        return active_precip / (0.1 * active_temps)
    return np.nan

def calculate_gtk_series(temp_ts: pd.Series, precip_ts: pd.Series) -> pd.Series:
    """
    Скользящий ГТК Селянинова (12-месячное окно).

    GTK(t) = sum(precip 12m) / (0.1 * sum(temp 12m where temp > 10°C))

    Возвращает непрерывный временной ряд.
    """

    # 12-месячные суммы осадков
    rolling_precip = precip_ts.rolling(window=12, min_periods=6).sum()

    # активная температура (только >10°C)
    active_temp = temp_ts.where(temp_ts > 10, 0)

    # 12-месячная сумма активных температур
    rolling_temp = active_temp.rolling(window=12, min_periods=6).sum()

    # избежание деления на 0
    denominator = 0.1 * rolling_temp.replace(0, np.nan)

    gtk = rolling_precip / denominator
    gtk.name = "gtk"

    return gtk

def graph_view(request):
    """Основная функция - все графики строятся здесь"""
    data_dir = os.path.join(settings.BASE_DIR, 'dmitrichenko', 'static', 'dmitrichenko', 'data')
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    selected_file = request.GET.get('file', files[0] if files else None)
    
    if not selected_file:
        return render(request, 'dmitrichenko/graph.html', {
            'files': files,
            'error': 'Нет доступных файлов данных'
        })
    
    temp_ts, precip_ts, df_long, lat = load_climate_timeseries(selected_file)
    ru_months_Title = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    # === КОНЕЦ СОЗДАНИЯ ВРЕМЕННЫХ РЯДОВ ===
    
    # --- ЛОГИКА ВЫБОРА ГОДА ИЗ GET-ЗАПРОСА ---
    available_years = sorted(df_long.index.year.unique().tolist())
    selected_year_str = request.GET.get('year')
    
    if selected_year_str and selected_year_str.isdigit() and int(selected_year_str) in available_years:
        target_year = int(selected_year_str)
    else:
        target_year = available_years[0]

    df_year = df_long[df_long.index.year == target_year]

    # --- 2. ПРОГНОЗ ARIMA (вызов отдельной функции) ---
    forecast_t, forecast_p = get_arima_forecast(temp_ts, precip_ts, selected_file)
    forecast_dates = pd.date_range(start=temp_ts.index[-1] + pd.offsets.MonthBegin(), periods=24, freq='MS')

    # ГЛАВНЫЙ ГРАФИК
    # Создаём 2 вертикальных графика с общей осью X
    fig_main = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Температура", "Осадки")
    )

    # Верхний график — температура
    fig_main.add_trace(
        go.Scatter(
            x=temp_ts.index,
            y=temp_ts,
            name='Температура',
            line=dict(color='red')
        ),
        row=1, col=1
    )

    fig_main.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_t,
            name='Прогноз температуры',
            line=dict(color='red', dash='dash')
        ),
        row=1, col=1
    )

    # Нижний график — осадки
    fig_main.add_trace(
        go.Scatter(
            x=precip_ts.index,
            y=precip_ts,
            name='Осадки',
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    fig_main.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=forecast_p,
            name='Прогноз осадков',
            line=dict(color='blue', dash='dash')
        ),
        row=2, col=1
    )

    # Оформление
    fig_main.update_layout(
        title=f'Метеорологический тренд и прогноз: {selected_file}',
        template='plotly_white',
        hovermode='x unified',
        height=700,

        # Одна общая нижняя шкала с ползунком
        xaxis2=dict(
            title='Год',
            rangeslider=dict(visible=True)
        ),

        showlegend=True
    )

    # Подписи осей Y
    fig_main.update_yaxes(title_text="Температура (°C)", row=1, col=1)
    fig_main.update_yaxes(title_text="Осадки (мм)", row=2, col=1)

    plot_div_main = pyo.plot(fig_main, output_type='div')

    # --- 3. ВЫЧИСЛЕНИЯ ГТК И АКТИВНЫХ ТЕМПЕРАТУР ---
    df_long['temp_active'] = df_long['temp'].apply(lambda x: x if x > 10 else 0)

    gtk = df_long.groupby(df_long.index.year).apply(calculate_gtk).dropna()
    current_gtk_value = round(gtk.iloc[-1], 2) if not gtk.empty else 0

    active_temp = df_long[(df_long.index.month >= 5) & (df_long.index.month <= 8)]
    active_sum = active_temp.groupby(active_temp.index.year)['temp_active'].sum()

# --- 4. ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ---
    df_year = df_long[df_long.index.year == target_year]

    # 1) Температура
    fig1_temp = go.Figure()
    fig1_temp.add_trace(go.Scatter(
        x=ru_months_Title,
        y=df_year['temp'],
        mode='lines+markers',
        name='Температура (°C)',
        line=dict(color='red')
    ))
    fig1_temp.update_layout(
        title=f'Температура за {target_year} год',
        xaxis_title='Месяцы',
        yaxis_title='Температура (°C)',
        template='plotly_white'
    )
    plot_div_1_temp = pyo.plot(fig1_temp, output_type='div')


    # 2) Осадки
    fig1_precip = go.Figure()
    fig1_precip.add_trace(go.Scatter(
        x=ru_months_Title,
        y=df_year['precip'],
        mode='lines+markers',
        name='Осадки (мм)',
        line=dict(color='blue')
    ))
    fig1_precip.update_layout(
        title=f'Осадки за {target_year} год',
        xaxis_title='Месяцы',
        yaxis_title='Осадки (мм)',
        template='plotly_white'
    )
    plot_div_1_precip = pyo.plot(fig1_precip, output_type='div')

    # 2) Температуры май-август
    df_year_active = df_year[(df_year.index.month >= 5) & (df_year.index.month <= 8)]
    active_sum_year = round(df_year_active["temp_active"].sum(), 1)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=['Май', 'Июн', 'Июл', 'Авг'], y=df_year_active['temp'], mode='lines+markers', name='Температура', line=dict(color='orange')))
    fig2.update_layout(title=f'Температуры (Май-Авг, {target_year}). Сумма активных: {active_sum_year}°C', xaxis_title='Месяцы', yaxis_title='Температура (°C)', template='plotly_white')
    plot_div_2 = pyo.plot(fig2, output_type='div')

    # 3) ГТК по годам
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=gtk.index, y=gtk.values, mode='lines+markers', name='ГТК', line=dict(color='green')))
    fig3.update_layout(title='Гидротермический коэффициент (ГТК) по годам', xaxis_title='Год', yaxis_title='Коэффициент', template='plotly_white')
    plot_div_3 = pyo.plot(fig3, output_type='div')

    # ==========================================
    # 4) РАЗДЕЛЕННЫЕ ГИСТОГРАММЫ ЗА ВЫБРАННЫЙ ГОД
    # ==========================================
    
    # 4.1) Гистограмма температур
    fig4_temp = go.Figure()
    fig4_temp.add_trace(go.Bar(x=ru_months_Title, y=df_year['temp'], name='Температура (°C)', marker_color='red'))
    fig4_temp.update_layout(title=f'Распределение температур ({target_year} год)', xaxis_title='Месяцы', yaxis_title='Температура (°C)', template='plotly_white')
    plot_div_4_temp = pyo.plot(fig4_temp, output_type='div')

    # 4.2) Гистограмма осадков
    fig4_precip = go.Figure()
    fig4_precip.add_trace(go.Bar(x=ru_months_Title, y=df_year['precip'], name='Осадки (мм)', marker_color='blue'))
    fig4_precip.update_layout(title=f'Распределение осадков ({target_year} год)', xaxis_title='Месяцы', yaxis_title='Осадки (мм)', template='plotly_white')
    plot_div_4_precip = pyo.plot(fig4_precip, output_type='div')

    # ==========================================
    # 5) РАЗДЕЛЕННЫЕ СРЕДНЕМНОГОЛЕТНИЕ ГИСТОГРАММЫ
    # ==========================================
    df_avg_month = df_long.groupby(df_long.index.month)[['temp', 'precip']].mean()
    
    # 5.1) Среднемноголетние температуры
    fig5_temp = go.Figure()
    fig5_temp.add_trace(go.Bar(x=ru_months_Title, y=df_avg_month['temp'], name='Темп. (ср)', marker_color='salmon'))
    fig5_temp.update_layout(title='Среднемноголетние значения температур', xaxis_title='Месяцы', yaxis_title='Температура (°C)', template='plotly_white')
    plot_div_5_temp = pyo.plot(fig5_temp, output_type='div')

    # 5.2) Среднемноголетние осадки
    fig5_precip = go.Figure()
    fig5_precip.add_trace(go.Bar(x=ru_months_Title, y=df_avg_month['precip'], name='Осадки (ср)', marker_color='lightblue'))
    fig5_precip.update_layout(title='Среднемноголетние значения осадков', xaxis_title='Месяцы', yaxis_title='Осадки (мм)', template='plotly_white')
    plot_div_5_precip = pyo.plot(fig5_precip, output_type='div')

    # 6) Сумма активных температур по годам
    fig6 = go.Figure()
    fig6.add_trace(go.Bar(x=active_sum.index, y=active_sum.values, name='Сумма акт. температур', marker_color='darkorange'))
    fig6.add_hline(y=active_sum.mean(), line_dash="dot", annotation_text="Среднемноголетнее", annotation_position="bottom right")
    fig6.update_layout(title='Сумма активных температур (Май - Август) по годам', xaxis_title='Год', yaxis_title='Сумма температур (°C)', template='plotly_white')
    plot_div_6 = pyo.plot(fig6, output_type='div')

    return render(request, 'dmitrichenko/graph.html', {
        'selected_file': selected_file,
        'files': files,
        'available_years': available_years,
        'target_year': target_year,
        'current_gtk': current_gtk_value,
        'plot_div_main': plot_div_main,

        'plot_div_1_temp': plot_div_1_temp,
        'plot_div_1_precip': plot_div_1_precip,

        'plot_div_2': plot_div_2,
        'plot_div_3': plot_div_3,
        'plot_div_4_temp': plot_div_4_temp,
        'plot_div_4_precip': plot_div_4_precip,
        'plot_div_5_temp': plot_div_5_temp,
        'plot_div_5_precip': plot_div_5_precip,
        'plot_div_6': plot_div_6,
    })
    
def get_graph() -> str:
    """Конвертирует текущую фигуру matplotlib в base64 строку."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    return image_base64

# ==========================================
# 1. Временной ряд индекса
# ==========================================
def plot_index_timeseries(series: pd.Series, 
                           index_name: str,
                           station_name: str = '') -> str:
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Цветовая заливка: положительные / отрицательные
    ax.fill_between(series.index, series, 0,
                    where=(series >= 0), 
                    color='steelblue', alpha=0.5, label='Влажный')
    ax.fill_between(series.index, series, 0,
                    where=(series < 0), 
                    color='tomato', alpha=0.5, label='Засушливый')
    
    ax.plot(series.index, series, color='black', linewidth=0.8, alpha=0.7)
    ax.axhline(0, color='black', linewidth=1)
    ax.axhline(1.0, color='green', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(-1.0, color='orange', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(2.0, color='darkblue', linewidth=0.8, linestyle=':', alpha=0.6)
    ax.axhline(-2.0, color='darkred', linewidth=0.8, linestyle=':', alpha=0.6)
    
    ax.set_title(f'{index_name} — {station_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel(index_name)
    ax.legend(loc='upper right', fontsize=10)
    
    # Форматирование дат
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return get_graph()

# ==========================================
# 2. ACF / PACF
# ==========================================
def plot_acf_pacf(series: pd.Series, lags: int = 40) -> tuple:
    """Возвращает два графика: ACF и PACF."""
    series_diff = series.diff().dropna()
    
    # ACF
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(series_diff, lags=lags, ax=ax, color='steelblue')
    ax.set_title('Функция автокорреляции (ACF) — первая разность', fontsize=13)
    ax.set_xlabel('Лаги')
    ax.set_ylabel('Автокорреляция')
    plt.tight_layout()
    acf_img = get_graph()
    
    # PACF
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_pacf(series_diff, lags=min(lags, len(series_diff)//2 - 1), 
              ax=ax, method='ywm', color='tomato')
    ax.set_title('Частичная функция автокорреляции (PACF) — первая разность', fontsize=13)
    ax.set_xlabel('Лаги')
    ax.set_ylabel('Частичная автокорреляция')
    plt.tight_layout()
    pacf_img = get_graph()
    
    return acf_img, pacf_img

# ==========================================
# 3. Декомпозиция ряда
# ==========================================
def plot_decomposition(series: pd.Series, index_name: str) -> str:
    clean = series.dropna()
    decomp = seasonal_decompose(clean, model='additive', period=12)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    axes[0].plot(decomp.observed, color='steelblue', linewidth=0.9)
    axes[0].set_title(f'Декомпозиция {index_name}', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Наблюдаемые')
    
    axes[1].plot(decomp.trend, color='darkorange', linewidth=0.9)
    axes[1].set_ylabel('Тренд')
    
    axes[2].plot(decomp.seasonal, color='green', linewidth=0.9)
    axes[2].set_ylabel('Сезонность')
    
    axes[3].plot(decomp.resid, color='gray', linewidth=0.9)
    axes[3].axhline(0, color='black', linewidth=0.8)
    axes[3].set_ylabel('Остатки')
    axes[3].set_xlabel('Дата')
    
    for ax in axes:
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return get_graph()

# ==========================================
# 4. Скользящее среднее и стандартное отклонение
# ==========================================
def plot_rolling_stats(series: pd.Series, 
                        index_name: str,
                        window: int = 12) -> str:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    ax1.plot(series, label='Исходный ряд', color='steelblue', alpha=0.7, linewidth=0.9)
    ax1.plot(rolling_mean, label=f'Скользящее среднее ({window}м)', 
             color='red', linewidth=2)
    ax1.fill_between(series.index, 
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.2, color='red', label='±σ')
    ax1.set_title(f'Скользящее среднее — {index_name}', fontsize=13, fontweight='bold')
    ax1.set_ylabel(index_name)
    ax1.legend()
    
    ax2.plot(rolling_std, color='darkorange', linewidth=1.5)
    ax2.set_title(f'Скользящее стандартное отклонение ({window} мес.)', fontsize=12)
    ax2.set_xlabel('Дата')
    ax2.set_ylabel('Стд. отклонение')
    
    plt.tight_layout()
    return get_graph()

# ==========================================
# 6. Тепловая карта (год × месяц)
# ==========================================
def plot_heatmap(series: pd.Series, index_name: str) -> str:
    df = pd.DataFrame({'value': series})
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    pivot = df.pivot_table(values='value', index='year', columns='month')
    pivot.columns = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                     'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.3)))
    
    sns.heatmap(pivot, ax=ax, cmap='RdBu', center=0,
                annot=False, fmt='.1f',
                linewidths=0.3, linecolor='gray',
                cbar_kws={'label': index_name})
    
    ax.set_title(f'Тепловая карта {index_name} (год × месяц)', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Месяц')
    ax.set_ylabel('Год')
    
    plt.tight_layout()
    return get_graph()

# ==========================================
# 7. Прогнозный график
# ==========================================
def plot_forecast(train: pd.Series, test: pd.Series,
                  results: dict, model_name: str,
                  index_name: str) -> str:
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Обучающий ряд
    ax.plot(train.index, train, 
            label='Обучающий ряд', color='steelblue', linewidth=1)
    
    # Fitted values
    if 'fitted_values' in results and results['fitted_values'] is not None:
        fv = results['fitted_values']
        ax.plot(fv.index, fv, 
                label='Подогнанные значения', 
                color='royalblue', linewidth=1, 
                linestyle='--', alpha=0.7)
    
    # Тестовый ряд
    ax.plot(test.index, test, 
            label='Реальные значения (тест)', 
            color='black', linewidth=1.5)
    
    # Прогноз
    ax.plot(results['forecast'].index, results['forecast'],
            label=f'Прогноз {model_name}', 
            color='tomato', linewidth=2, marker='o', markersize=3)
    
    # Разделительная линия
    ax.axvline(x=test.index[0], color='gray', linestyle=':', linewidth=1.5,
               label='Начало прогноза')
    
    ax.set_title(f'Прогноз {model_name} — {index_name}', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Дата')
    ax.set_ylabel(index_name)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return get_graph()

def lab2_view(request):
    data_dir = os.path.join(settings.BASE_DIR, 'dmitrichenko', 'static', 'dmitrichenko', 'data')
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    selected_file = request.GET.get('file', files[0] if files else None)
    
    if not selected_file:
        return render(request, 'dmitrichenko/lab2_results.html', {
            'files': files,
            'error': 'Нет доступных файлов данных'
        })
    
    temp_ts, precip_ts, df_long, lat = load_climate_timeseries(selected_file)
    ru_months_Title = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    # === КОНЕЦ СОЗДАНИЯ ВРЕМЕННЫХ РЯДОВ ===
    
    # Температурная аппроксимация
    tmean = temp_ts
    tmax = temp_ts + 5
    tmin = temp_ts - 5

    # PET (потенциальная испаряемость)
    pet = pyet.hargreaves(tmean, tmax, tmin, lat=lat)

    # Водный баланс для SPEI
    water_balance = precip_ts - pet

    # -----------------------------
    # Расчёт индексов
    # -----------------------------

    all_indices = {
        'spi3': si.spi(precip_ts, timescale=3),
        'spei3': si.spei(water_balance, timescale=3),
        'spei6': si.spei(water_balance, timescale=6),
        'gtk': calculate_gtk_series(temp_ts, precip_ts),
        'temp': temp_ts,
        'precip': precip_ts
    }
    
    indices_map = {
        'spi3': 'SPI-3',
        'spei3': 'SPEI-3',
        'spei6': 'SPEI-6',
        'gtk': 'ГТК',
        'temp': 'Температура',
        'precip': 'Осадки'
    }
    selected_index_key = request.GET.get('index', 'spei3')
    
    full_series = all_indices[selected_index_key].dropna()
    index_display_name = indices_map[selected_index_key]
    first_diff = full_series.diff().dropna() # Первая разность для графиков

    # --- ЛОГИКА ПРОГНОЗА ---
    # Разделяем на train и test (например, последние 12 месяцев на тест)
    split_idx = int(len(full_series) * 0.85)
    train_set = full_series.iloc[:split_idx]
    test_set = full_series.iloc[split_idx:]

    # Обучаем модель на тренировочных данных
    model_arima = ARIMA(train_set, order=(12, 0, 0))
    fitted_model = model_arima.fit()

    # Делаем прогноз на длину тестовой выборки
    forecast_steps = len(test_set)
    forecast_values = fitted_model.get_forecast(steps=forecast_steps).predicted_mean
    
    # Подготавливаем словарь для функции plot_forecast
    forecast_results = {
        'forecast': forecast_values,
        'fitted_values': fitted_model.fittedvalues
    }

    # Генерируем график прогноза
    forecast_plot = plot_forecast(
        train_set, 
        test_set, 
        forecast_results, 
        "ARIMA", 
        index_display_name
    )

    # ==========================================
    # ГЕНЕРАЦИЯ ЭСТЕТИЧНЫХ ГРАФИКОВ
    # ==========================================
    # 1. Основной график ряда
    ts_plot = plot_index_timeseries(full_series, index_display_name, selected_file)
    
    # 2. ACF / PACF (возвращает кортеж из двух строк base64)
    acf_img, pacf_img = plot_acf_pacf(full_series)
    
    # 3. Декомпозиция
    decomp_plot = plot_decomposition(full_series, index_display_name)
    
    # 4. Скользящие показатели
    rolling_plot = plot_rolling_stats(full_series, index_display_name)
    
    # 5. Тепловая карта
    heatmap_plot = plot_heatmap(full_series, index_display_name)

    # ==========================================
    # СТАТИСТИЧЕСКИЕ ТЕСТЫ (ADF и KPSS)
    # ==========================================
    # ADF Test
    adf_test = sm.tsa.adfuller(full_series)
    adf_pvalue = adf_test[1]
    adf_result = "Stationary (Стационарен)" if adf_pvalue < 0.05 else "Non-stationary (Нестационарен)"

    # KPSS Test
    kpss_test = sm.tsa.kpss(full_series, nlags="auto")
    kpss_pvalue = kpss_test[1]
    kpss_result = "Non-stationary (Нестационарен)" if kpss_pvalue < 0.05 else "Stationary (Стационарен)"

    # ==========================================
    # ГРАФИКИ ACF и PACF
    # ==========================================
    # ACF
    plt.figure(figsize=(10, 6))
    plot_acf(first_diff, lags=40, ax=plt.gca()) # ax=plt.gca() важен для правильной отрисовки в Django
    plt.title('Autocorrelation Function (ACF)', fontsize=14)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Autocorrelation', fontsize=12)
    plt.grid()
    acf_graph = get_graph()

    # PACF
    plt.figure(figsize=(10, 6))
    plot_pacf(first_diff, lags=40, ax=plt.gca(), method='ywm')
    plt.title('Partial Autocorrelation Function (PACF)', fontsize=14)
    plt.xlabel('Lags', fontsize=12)
    plt.ylabel('Partial Autocorrelation', fontsize=12)
    plt.grid()
    pacf_graph = get_graph()

    # ==========================================
    # МОДЕЛЬ ARIMA И ДИАГНОСТИКА
    # ==========================================
    model = ARIMA(full_series, order=(12, 0, 0))
    fitted_model = model.fit()
    
    # Сохраняем summary в виде HTML-таблицы для красивого вывода
    arima_summary = fitted_model.summary().as_html()

    # Графики диагностики
    fitted_model.plot_diagnostics(figsize=(10, 6))
    diagnostics_graph = get_graph()

    # ==========================================
    # ПЕРЕДАЧА ДАННЫХ В ШАБЛОН
    # ==========================================
    context = {
        'files': files,
        'selected_file': selected_file,
        'indices': indices_map,
        'selected_index': selected_index_key,
        
        'ts_plot': ts_plot,
        'acf_graph': acf_img,
        'pacf_graph': pacf_img,
        'decomp_plot': decomp_plot,
        'rolling_plot': rolling_plot,
        'heatmap_plot': heatmap_plot,
        'forecast_plot': forecast_plot,
        
        'adf_pvalue': round(float(adf_test[1]), 6),
        'adf_result': "Стационарен" if adf_test[1] < 0.05 else "Нестационарен",
        'kpss_pvalue': round(float(kpss_test[1]), 6),
        'kpss_result': "Нестационарен" if kpss_test[1] < 0.05 else "Стационарен",
        
        'arima_summary': arima_summary,
        'diagnostics_graph': diagnostics_graph,
    }

    return render(request, 'dmitrichenko/lab2_results.html', context)