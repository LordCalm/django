import os
import csv
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
import pmdarima as pm
from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from django.core.cache import cache
from .forms import UserForm

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
    Отдельная функция для расчета ARIMA прогноза температуры и осадков.
    Результат кэшируется на 1 час.
    """
    cache_key = f"forecast_{selected_file}"
    cached = cache.get(cache_key)
    
    if cached:
        return cached
    
    # Модель для температуры
    model_t = pm.auto_arima(
        temp_ts,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    forecast_t = model_t.predict(n_periods=24)
    
    # Модель для осадков
    model_p = pm.auto_arima(
        precip_ts,
        seasonal=True,
        m=12,
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore"
    )
    forecast_p = model_p.predict(n_periods=24)
    
    cache.set(cache_key, (forecast_t, forecast_p), timeout=3600)
    
    return forecast_t, forecast_p


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
    
    file_path = os.path.join(data_dir, selected_file)
    df = pd.read_csv(file_path, sep=';', encoding='cp1251')
    
    # Заменяем заглушки на NaN
    df = df.replace([999.9, -999.9, '999.9', '-999.9'], np.nan)
    df = df.dropna(subset=['год'])

    # --- 1. ПРЕОБРАЗОВАНИЕ В ПОМЕСЯЧНЫЙ РЯД ---
    months = ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек']
    ru_months_Title = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
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
    
    # === ДОПОЛНИТЕЛЬНОЕ ПРЕОБРАЗОВАНИЕ В ЧИСЛА (на всякий случай) ===
    df_long['temp'] = pd.to_numeric(df_long['temp'], errors='coerce')
    df_long['precip'] = pd.to_numeric(df_long['precip'], errors='coerce')
    
    # === ЗАМЕНА NaN НА МЕДИАНЫ ПО МЕСЯЦАМ ДЛЯ ТЕМПЕРАТУРЫ И ОСАДКОВ ===
    # Вычисляем медианы для каждого месяца
    monthly_temp_median = df_long.groupby(df_long.index.month)['temp'].median()
    monthly_precip_median = df_long.groupby(df_long.index.month)['precip'].median()
    
    # Заменяем NaN на медианы по соответствующим месяцам
    df_long['temp'] = df_long.apply(
        lambda row: monthly_temp_median[row.name.month] if pd.isna(row['temp']) else row['temp'], 
        axis=1
    )
    df_long['precip'] = df_long.apply(
        lambda row: monthly_precip_median[row.name.month] if pd.isna(row['precip']) else row['precip'], 
        axis=1
    )
    
    # Финальная обработка оставшихся NaN (если медиана тоже была NaN)
    df_long['temp'] = df_long['temp'].fillna(df_long['temp'].median())
    df_long['precip'] = df_long['precip'].fillna(df_long['precip'].median())

    # === СОЗДАНИЕ ВРЕМЕННЫХ РЯДОВ С ЗАПОЛНЕНИЕМ ПРОПУСКОВ ===
    temp_ts = df_long['temp'].astype(float).asfreq('MS')
    precip_ts = df_long['precip'].astype(float).asfreq('MS')
    
    # Заполняем пропуски, возникшие после asfreq, медианами по месяцам
    for month in range(1, 13):
        month_mask = temp_ts.index.month == month
        temp_median = monthly_temp_median.get(month, temp_ts.median())
        precip_median = monthly_precip_median.get(month, precip_ts.median())
        
        temp_ts = temp_ts.fillna(temp_ts.where(~month_mask).fillna(temp_median))
        precip_ts = precip_ts.fillna(precip_ts.where(~month_mask).fillna(precip_median))
    
    # Дополнительная интерполяция и заполнение
    temp_ts = temp_ts.interpolate(method='linear', limit=2).bfill().ffill()
    precip_ts = precip_ts.interpolate(method='linear', limit=2).bfill().ffill()
    
    # Финальная проверка - заменяем оставшиеся NaN
    temp_ts = temp_ts.fillna(temp_ts.median())
    precip_ts = precip_ts.fillna(precip_ts.median())
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
    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=temp_ts.index, y=temp_ts, name='Температура', line=dict(color='red')))
    fig_main.add_trace(go.Scatter(x=forecast_dates, y=forecast_t, name='Прогноз температуры', line=dict(dash='dash', color='red')))
    fig_main.add_trace(go.Scatter(x=precip_ts.index, y=precip_ts, name='Осадки', line=dict(color='blue'), yaxis='y2'))
    fig_main.add_trace(go.Scatter(x=forecast_dates, y=forecast_p, name='Прогноз осадков', line=dict(dash='dash', color='blue'), yaxis='y2'))
    fig_main.update_layout(
        title=f'Метеорологический тренд и прогноз: {selected_file}',
        xaxis_title='Год', yaxis_title='Температура (°C)',
        yaxis2=dict(title='Осадки (мм)', overlaying='y', side='right'),
        template='plotly_white', hovermode="x unified",
        xaxis=dict(rangeslider=dict(visible=True)) 
    )
    plot_div_main = pyo.plot(fig_main, output_type='div')

    # --- 3. ВЫЧИСЛЕНИЯ ГТК И АКТИВНЫХ ТЕМПЕРАТУР ---
    df_long['temp_active'] = df_long['temp'].apply(lambda x: x if x > 10 else 0)

    def calculate_gtk(group):
        active_mask = group['temp'] > 10
        active_temps = group.loc[active_mask, 'temp'].sum()
        active_precip = group.loc[active_mask, 'precip'].sum()
        if active_temps > 0:
            return active_precip / (0.1 * active_temps)
        return None

    gtk = df_long.groupby(df_long.index.year).apply(calculate_gtk).dropna()
    current_gtk_value = round(gtk.iloc[-1], 2) if not gtk.empty else 0

    active_temp = df_long[(df_long.index.month >= 5) & (df_long.index.month <= 8)]
    active_sum = active_temp.groupby(active_temp.index.year)['temp_active'].sum()

    # --- 4. ДОПОЛНИТЕЛЬНЫЕ ГРАФИКИ ---
    df_year = df_long[df_long.index.year == target_year]

    # 1) Линейный график: осадки и температура на год
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ru_months_Title, y=df_year['temp'], mode='lines+markers', name='Температура (°C)', line=dict(color='red')))
    fig1.add_trace(go.Scatter(x=ru_months_Title, y=df_year['precip'], mode='lines+markers', name='Осадки (мм)', yaxis='y2', line=dict(color='blue')))
    fig1.update_layout(title=f'Осадки и температуры за {target_year} год', xaxis_title='Месяцы', yaxis_title='Температура (°C)', yaxis2=dict(title='Осадки (мм)', overlaying='y', side='right'), template='plotly_white')
    plot_div_1 = pyo.plot(fig1, output_type='div')

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

    # 4) Гистограмма осадков и температур
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=ru_months_Title, y=df_year['temp'], name='Температура (°C)', marker_color='red'))
    fig4.add_trace(go.Bar(x=ru_months_Title, y=df_year['precip'], name='Осадки (мм)', yaxis='y2', marker_color='blue'))
    fig4.update_layout(title=f'Распределение осадков и температур ({target_year} год)', xaxis_title='Месяцы', yaxis_title='Температура (°C)', yaxis2=dict(title='Осадки (мм)', overlaying='y', side='right'), barmode='group', template='plotly_white')
    plot_div_4 = pyo.plot(fig4, output_type='div')

    # 5) Среднемноголетние
    df_avg_month = df_long.groupby(df_long.index.month)[['temp', 'precip']].mean()
    fig5 = go.Figure()
    fig5.add_trace(go.Bar(x=ru_months_Title, y=df_avg_month['temp'], name='Темп. (ср)', marker_color='salmon'))
    fig5.add_trace(go.Bar(x=ru_months_Title, y=df_avg_month['precip'], name='Осадки (ср)', yaxis='y2', marker_color='lightblue'))
    fig5.update_layout(title='Среднемноголетние значения осадков и температур', xaxis_title='Месяцы', yaxis_title='Температура (°C)', yaxis2=dict(title='Осадки (мм)', overlaying='y', side='right'), barmode='group', template='plotly_white')
    plot_div_5 = pyo.plot(fig5, output_type='div')

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
        'plot_div_1': plot_div_1,
        'plot_div_2': plot_div_2,
        'plot_div_3': plot_div_3,
        'plot_div_4': plot_div_4,
        'plot_div_5': plot_div_5,
        'plot_div_6': plot_div_6,
    })