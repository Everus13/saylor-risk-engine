import streamlit as st
import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Обеспечиваем нахождение корня проекта в python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import DEBT_SCHEDULE_PATH, LIGHTGBM_MODEL_PATH, PYTORCH_MODEL_PATH, RL_MODEL_PATH
from src.financial_tracker import FinancialTracker
from src.order_book_sim import OrderBookSimulator
from src.ml_impact import PriceImpactMLPipeline, generate_ml_dataset
from src.rl_agent.train import RLExecutionTrainer, SB3_AVAILABLE
from src.utils import ensure_models_exist, plot_order_book_chart

# --- Инициализация страницы и стили ---
st.set_page_config(
    page_title="MSTR-BTC Impact & Execution Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Премиум стилизация
st.markdown("""
<style>
    .main {
        background-color: #0d0f12;
        color: #e2e8f0;
    }
    .stCard {
        background-color: #1a1f26 !important;
        border: 1px solid #2d3748 !important;
        padding: 20px !important;
        border-radius: 10px !important;
        margin-bottom: 20px !important;
    }
    .metric-value {
        font-size: 32px !important;
        font-weight: 700 !important;
        color: #3182ce !important;
    }
    .metric-label {
        font-size: 14px !important;
        color: #a0aec0 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #3182ce;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1f26;
        border-radius: 4px 4px 0px 0px;
        color: #cbd5e0;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3182ce !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# Убедимся, что модели инициализированы перед стартом
ensure_models_exist()

# Создаем объекты бизнес-логики
tracker = FinancialTracker()
ml_pipeline = PriceImpactMLPipeline()
rl_trainer = RLExecutionTrainer()

# --- Кэширование сетевых запросов к SEC EDGAR ---
@st.cache_data(ttl=1800)
def get_cached_sec_facts():
    return tracker.fetch_sec_edgar_facts()

# --- Боковая панель настроек (Sidebar) ---
st.sidebar.title("🛠 Настройки системы")

# Получение живых цен через yfinance
with st.sidebar.spinner("Получение текущих цен..."):
    live_prices = tracker.get_current_prices()

btc_price_input = st.sidebar.number_input(
    "Текущий курс BTC ($)", 
    min_value=1000.0, 
    value=float(live_prices["BTC"]), 
    step=500.0,
    help="Курс Биткоина в реальном времени. По умолчанию загружается из Yahoo Finance."
)

mstr_price_input = st.sidebar.number_input(
    "Цена акции MSTR ($)", 
    min_value=1.0, 
    value=float(live_prices["MSTR"]), 
    step=5.0
)

cash_reserve_input = st.sidebar.slider(
    "Долларовый резерв кэша MSTR ($ млн)", 
    min_value=0.0, 
    max_value=1000.0, 
    value=float(tracker.data.get("usd_cash_reserve", 120000000.0) / 1e6),
    step=10.0
) * 1e6

# Обновляем резерв кэша в модели
tracker.update_cash_reserve(cash_reserve_input)

# Синхронизация с SEC EDGAR
use_sec_debt = st.sidebar.checkbox(
    "Синхронизация с SEC EDGAR", 
    value=True, 
    help="Если включено, общий объем долга и баланс будут синхронизированы с официальными данными SEC в реальном времени."
)

# --- Главный Заголовок ---
st.title("MSTR-BTC Impact & Execution Engine")
st.markdown("Система предиктивной аналитики, анализа влияния ордеров на ликвидность рынка и планирования оптимального исполнения для MicroStrategy.")

# Вкладки панели управления
tab_debt, tab_impact, tab_rl = st.tabs([
    "📅 Календарь долгов и резервов", 
    "📈 Симулятор импакта стакана L2", 
    "🤖 Оптимальное исполнение (RL)"
])

# ==============================================================================
# ВКЛАДКА 1: КАЛЕНДАРЬ ДОЛГОВ И РЕЗЕРВОВ
# ==============================================================================
with tab_debt:
    st.header("Календарь обязательств и оценка потребности в ликвидности")
    
    # Секция данных SEC EDGAR
    st.subheader("📊 Данные баланса из SEC EDGAR (в реальном времени)")
    with st.spinner("Загрузка отчетов SEC EDGAR..."):
        sec_facts = get_cached_sec_facts()
        
    c_sec1, c_sec2, c_sec3, c_sec4 = st.columns(4)
    with c_sec1:
        st.metric(
            label="Общий долгосрочный долг (SEC)", 
            value=f"${sec_facts.get('long_term_debt', 0.0):,.0f}",
            help="Общая сумма долгосрочного долга, отраженная на балансе компании по форме 10-K/10-Q."
        )
    with c_sec2:
        st.metric(
            label="Краткосрочный долг (SEC)", 
            value=f"${sec_facts.get('long_term_debt_current', 0.0):,.0f}"
        )
    with c_sec3:
        st.metric(
            label="Привилегированные акции (SEC)", 
            value=f"${sec_facts.get('preferred_stock_value', 0.0):,.0f}"
        )
    with c_sec4:
        st.metric(
            label="Дата последнего отчета SEC", 
            value=sec_facts.get("date", "Н/Д"),
            delta=f"Форма: {sec_facts.get('form', 'Н/Д')}",
            delta_color="normal"
        )
        
    st.markdown("---")
    
    # Входные параметры прогноза
    col_input, col_space = st.columns([1, 3])
    with col_input:
        forecast_days = st.selectbox("Горизонт прогнозирования выплат", [30, 90, 180, 360], index=1)
        
    # Расчет потребности
    reqs = tracker.get_btc_sell_requirements_sync(
        days_forecast=forecast_days, 
        btc_price=btc_price_input,
        use_sec_debt=use_sec_debt
    )
    
    # Карточки финансовых показателей
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Всего обязательств к выплате",
            value=f"${reqs['total_usd_required']:,.2f}",
            delta=f"Расходы Opex: ${reqs['breakdown']['total_opex_usd']:,.2f}",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="Долларовые резервы MSTR",
            value=f"${reqs['usd_cash_reserve']:,.2f}"
        )
    with col3:
        st.metric(
            label="Чистый дефицит фиата",
            value=f"${reqs['net_usd_needed']:,.2f}"
        )
    with col4:
        st.metric(
            label="BTC к продаже (Стресс-сценарий)",
            value=f"{reqs['btc_to_sell_stress_case']:.2f} BTC",
            delta="Базовый сценарий (ATM): 0.00 BTC",
            delta_color="inverse"
        )
        
    st.warning("⚠️ **Стресс-сценарий** предполагает полную приостановку или невозможность допэмиссии акций MSTR (программы ATM), что вынуждает компанию ликвидировать резервы Биткоина для покрытия текущих платежей.")

    # Таблица выплат и график
    breakdown_data = reqs["breakdown"]["payments_breakdown"]
    df_pmts = pd.DataFrame(breakdown_data)
    
    if len(df_pmts) > 0:
        df_pmts["date"] = pd.to_datetime(df_pmts["date"])
        df_pmts = df_pmts.sort_values(by="date")
        
        c_left, c_right = st.columns([1, 1])
        
        with c_left:
            st.subheader("📋 Список ближайших обязательств в USD")
            df_display = df_pmts.copy()
            df_display["date"] = df_display["date"].dt.strftime('%Y-%m-%d')
            df_display["amount_usd"] = df_display["amount_usd"].map(lambda x: f"${x:,.2f}")
            df_display.columns = ["Дата", "Тип обязательства", "Наименование инструмента", "Сумма (USD)"]
            st.dataframe(df_display, use_container_width=True)
            
        with c_right:
            st.subheader("📅 График распределения платежей по времени")
            fig_timeline = px.bar(
                df_pmts,
                x="date",
                y="amount_usd",
                color="type",
                hover_data=["name"],
                labels={"amount_usd": "Сумма (USD)", "date": "Дата платежа", "type": "Категория"},
                template="plotly_dark",
                title="Отток денежных средств по периодам"
            )
            fig_timeline.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_timeline, use_container_width=True)
    else:
        st.info("Нет плановых купонных или дивидендных выплат в выбранном горизонте прогнозирования.")

    # Базовая таблица инструментов
    st.subheader("🔗 Действующие долговые инструменты MicroStrategy (Справочно)")
    df_notes = pd.DataFrame(tracker.data.get("convertible_notes", []))
    if len(df_notes) > 0:
        df_notes = df_notes.copy()
        df_notes["principal"] = df_notes["principal"].map(lambda x: f"${x:,.2f}")
        df_notes["coupon_rate"] = df_notes["coupon_rate"].map(lambda x: f"{x * 100:.3f}%")
        df_notes.columns = ["Наименование выпуска", "Номинал ($)", "Процентная ставка", "Месяцы выплат купонов", "День выплаты", "Год погашения"]
        st.dataframe(df_notes, use_container_width=True)

# ==============================================================================
# ВКЛАДКА 2: СИМУЛЯТОР ИМПАКТА СТАКАНА L2
# ==============================================================================
with tab_impact:
    st.header("Анализ мгновенной ликвидности и ценового влияния")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Параметры рыночной продажи")
        sell_qty = st.slider("Объем продажи (BTC)", min_value=1.0, max_value=1000.0, value=150.0, step=5.0)
        
        exchange_select = st.selectbox("Источник стакана ордеров", ["Бинанс (Синтетическая глубина)", "CCXT Binance Live L2"])
        
        # Получение L2 стакана
        if exchange_select == "CCXT Binance Live L2":
            with st.spinner("Загрузка стакана Binance L2..."):
                book = OrderBookSimulator.get_live_order_book("binance", "BTC/USDT")
            if book is None:
                st.error("Не удалось получить стакан. Используем синтетическую модель.")
                book = OrderBookSimulator.generate_synthetic_order_book(mid_price=btc_price_input)
        else:
            book = OrderBookSimulator.generate_synthetic_order_book(mid_price=btc_price_input)
            
        # Симуляция прямого маркет-селла (L2 Walk)
        sim_res = OrderBookSimulator.simulate_market_sell(book, sell_qty)
        
        # Прогнозы ML-моделей
        lgb_preds = ml_pipeline.predict_impact(book, sell_qty, model_type="lightgbm")
        torch_preds = ml_pipeline.predict_impact(book, sell_qty, model_type="pytorch")
        
        st.markdown("### 📊 Показатели симуляции стакана")
        st.metric(
            label="Фактическое проскальзывание (VWAP)", 
            value=f"{sim_res['slippage_pct']:.4f}%",
            delta=f"Средняя цена продажи: ${sim_res['vwap']:,.2f}"
        )
        st.metric(
            label="Предельное падение цены (Marginal Impact)", 
            value=f"{sim_res['price_impact_pct']:.4f}%",
            delta=f"Худшая цена исполнения: ${sim_res['marginal_price']:,.2f}"
        )
        
    with col2:
        st.subheader("🎯 Прогнозные доверительные интервалы падения цены (Квантили)")
        
        # Сравнение прогнозов моделей по квантилям
        pred_comparison = pd.DataFrame({
            "Сценарий падения цены": ["Минимальный импакт (Квантиль 10%)", "Медианный импакт (Квантиль 50%)", "Худший импакт (Квантиль 90%)"],
            "Модель LightGBM": [f"-{lgb_preds[0.1]:.4f}%", f"-{lgb_preds[0.5]:.4f}%", f"-{lgb_preds[0.9]:.4f}%"],
            "Нейросеть PyTorch MLP": [f"-{torch_preds[0.1]:.4f}%", f"-{torch_preds[0.5]:.4f}%", f"-{torch_preds[0.9]:.4f}%"]
        })
        
        st.table(pred_comparison)
        
        # Визуализация доверительных интервалов
        fig_ci = go.Figure()
        # LightGBM Range
        fig_ci.add_trace(go.Bar(
            name="Диапазон LightGBM (10%-90%)",
            x=["LightGBM"],
            y=[lgb_preds[0.9] - lgb_preds[0.1]],
            base=lgb_preds[0.1],
            marker_color='rgba(49, 130, 206, 0.6)',
            width=0.4
        ))
        # PyTorch Range
        fig_ci.add_trace(go.Bar(
            name="Диапазон PyTorch MLP (10%-90%)",
            x=["PyTorch MLP"],
            y=[torch_preds[0.9] - torch_preds[0.1]],
            base=torch_preds[0.1],
            marker_color='rgba(159, 122, 234, 0.6)',
            width=0.4
        ))
        # Actual Line
        fig_ci.add_trace(go.Scatter(
            x=["LightGBM", "PyTorch MLP"],
            y=[sim_res["price_impact_pct"], sim_res["price_impact_pct"]],
            mode='markers',
            name='Фактический импакт по L2',
            marker=dict(color='yellow', size=12, symbol="star-diamond")
        ))
        
        fig_ci.update_layout(
            title=f"Прогнозируемые диапазоны падения цены при продаже {sell_qty} BTC (Доверительный интервал 80%)",
            yaxis_title="Падение цены в %",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_ci, use_container_width=True)
        
    st.subheader("📈 График плотности ликвидности и зона маркет-импакта")
    # Преобразование названий легенды графика в русский
    fig_ob = plot_order_book_chart(book, sell_qty)
    fig_ob.update_layout(
        title="Глубина стакана L2 и зона агрессивного влияния ордера",
        xaxis_title="Цена BTC (USD)",
        yaxis_title="Кумулятивный объем (BTC)"
    )
    # Корректируем названия графиков в plotly
    for trace in fig_ob.data:
        if trace.name == "Bids (Buyers)":
            trace.name = "Покупки (Bids)"
        elif trace.name == "Asks (Sellers)":
            trace.name = "Продажи (Asks)"
    st.plotly_chart(fig_ob, use_container_width=True)

    # Триггер переобучения моделей
    st.subheader("🧠 Оптимизация гиперпараметров и обучение моделей")
    if st.button("Запустить кросс-валидацию и Grid Search"):
        with st.spinner("Генерация расширенного датасета стаканов, кросс-валидация и оптимизация гиперпараметров..."):
            try:
                X, y = generate_ml_dataset(n_snapshots=150)
                split = int(len(X) * 0.8)
                X_train, X_val = X[:split], X[split:]
                y_train, y_val = y[:split], y[split:]
                
                pipeline = PriceImpactMLPipeline()
                pipeline.train_lightgbm_quantiles(X_train, y_train, X_val, y_val)
                pipeline.train_pytorch_quantiles(X_train, y_train, X_val, y_val)
                st.success("Обучение завершено. Метрики моделей обновлены!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Ошибка в процессе оптимизации: {e}")

# ==============================================================================
# ВКЛАДКА 3: ОПТИМАЛЬНОЕ ИСПОЛНЕНИЕ (RL)
# ==============================================================================
with tab_rl:
    st.header("Моделирование оптимального исполнения крупных ордеров")
    
    if not SB3_AVAILABLE:
        st.warning("⚠️ **stable-baselines3** не установлена в окружении. По умолчанию симуляция работает по алгоритму TWAP.")
        
    col_l, col_r = st.columns([1, 2])
    
    with col_l:
        st.subheader("Настройки симуляции исполнения")
        exec_volume = st.number_input("Общий объем продажи (BTC)", min_value=10.0, max_value=2000.0, value=300.0, step=50.0)
        exec_steps = st.slider("Временные интервалы (шаги исполнения)", min_value=5, max_value=30, value=15)
        
        strategy_options = ["twap", "rl"]
        selected_strat = st.selectbox(
            "Стратегия исполнения", 
            strategy_options, 
            format_func=lambda x: "Равномерное капание (TWAP)" if x == "twap" else "Обучение с подкреплением (RL PPO Агент)"
        )
        
        rl_exists = os.path.exists(RL_MODEL_PATH + ".zip") or os.path.exists(RL_MODEL_PATH)
        if selected_strat == "rl" and not rl_exists:
            st.info("Обученная модель PPO не найдена. Пожалуйста, запустите обучение агента ниже перед симуляцией.")
            
        run_sim = st.button("Запустить симуляцию ордера")
        
        st.markdown("---")
        st.subheader("Обучение RL-агента")
        timesteps_input = st.number_input("Количество шагов обучения (Epochs/Timesteps)", min_value=1000, max_value=100000, value=10000, step=5000)
        
        if st.button("Начать обучение RL PPO Агента"):
            with st.spinner("Запущен процесс обучения PPO в Gymnasium среде..."):
                try:
                    rl_trainer.train_agent(
                        total_timesteps=int(timesteps_input),
                        total_volume=exec_volume,
                        total_steps=exec_steps,
                        mid_price=btc_price_input
                    )
                    st.success("RL-агент успешно обучен и сохранен!")
                except Exception as e:
                    st.error(f"Не удалось обучить агента: {e}")
                    
    with col_r:
        if run_sim:
            with st.spinner("Симуляция выполнения стратегии..."):
                try:
                    # Запуск выбранной стратегии
                    df_strat, metrics_strat = rl_trainer.run_simulation(
                        total_volume=exec_volume,
                        total_steps=exec_steps,
                        starting_mid_price=btc_price_input,
                        strategy=selected_strat
                    )
                    
                    # Запуск TWAP для сравнения
                    df_twap, metrics_twap = rl_trainer.run_simulation(
                        total_volume=exec_volume,
                        total_steps=exec_steps,
                        starting_mid_price=metrics_strat["arrival_price"],
                        strategy="twap"
                    )
                    
                    st.subheader("Сводные результаты исполнения")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(
                            label="Средняя цена продажи (VWAP)", 
                            value=f"${metrics_strat['avg_execution_price']:,.2f}",
                            delta=f"TWAP: ${metrics_twap['avg_execution_price']:,.2f}"
                        )
                    with col_m2:
                        st.metric(
                            label="Общая полученная выручка", 
                            value=f"${metrics_strat['total_revenue_usd']:,.2f}",
                            delta=f"TWAP: ${metrics_twap['total_revenue_usd']:,.2f}"
                        )
                    with col_m3:
                        st.metric(
                            label="Среднее проскальзывание (Slippage)", 
                            value=f"{metrics_strat['total_slippage_pct']:.4f}%",
                            delta=f"TWAP: {metrics_twap['total_slippage_pct']:.4f}%",
                            delta_color="inverse"
                        )
                    
                    # Графики результатов
                    st.subheader("Динамика снижения объема (Траектория инвентаря)")
                    fig_trajectory = go.Figure()
                    fig_trajectory.add_trace(go.Scatter(
                        x=df_twap["step"], y=df_twap["remaining_volume"],
                        mode='lines+markers', name='Равномерный TWAP', line=dict(color='gray', dash='dash')
                    ))
                    fig_trajectory.add_trace(go.Scatter(
                        x=df_strat["step"], y=df_strat["remaining_volume"],
                        mode='lines+markers', name=f'Стратегия {selected_strat.upper()}', line=dict(color='#3182ce', width=3)
                    ))
                    fig_trajectory.update_layout(
                        xaxis_title="Временной шаг",
                        yaxis_title="Оставшийся объем BTC",
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_trajectory, use_container_width=True)
                    
                    st.subheader("Объемы продаж на каждом шаге (Ордера)")
                    fig_actions = go.Figure()
                    fig_actions.add_trace(go.Bar(
                        x=df_twap["step"], y=df_twap["filled_qty"],
                        name='Объем TWAP', marker_color='gray', opacity=0.6
                    ))
                    fig_actions.add_trace(go.Bar(
                        x=df_strat["step"], y=df_strat["filled_qty"],
                        name=f'Объем {selected_strat.upper()}', marker_color='#3182ce'
                    ))
                    fig_actions.update_layout(
                        xaxis_title="Временной шаг",
                        yaxis_title="Проданный объем BTC",
                        barmode='group',
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig_actions, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка при моделировании исполнения ордера: {e}")
        else:
            st.info("Задайте параметры исполнения на левой панели и нажмите 'Запустить симуляцию ордера', чтобы построить графики.")
