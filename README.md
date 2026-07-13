# MSTR-BTC Impact & Execution Engine (Strategy.com Style)

Система предиктивной аналитики и оптимального исполнения (Optimal Execution) для расчета долговых обязательств MicroStrategy (MSTR) и оптимизации продаж биткоина с использованием машинного обучения (Quantile Regression) и глубокого обучения с подкреплением (RL PPO).

---

## 🛠 Новая структура проекта

Проект переведен на клиент-серверную архитектуру:
*   **[backend/](file:///c:/Users/111/.gemini/antigravity-ide/scratch/babki_project/backend)** — Веб-бэкенд на **FastAPI**:
    *   `main.py` — Точка входа API и WebSocket-сервер для стриминга логов обучения RL.
    *   `src/database.py` — Персистентное хранилище на SQLite для настроек и истории симуляций.
    *   `src/financial_tracker.py` — Динамический парсинг SEC EDGAR и каскадный сборщик цен BTC/MSTR.
    *   `src/order_book_sim.py` — Оптимизированный симулятор стакана на NumPy.
    *   `src/ml_impact.py` — Модели оценки импакта (PyTorch Deep MLP и LightGBM).
    *   `src/rl_agent/` — Окружение Gymnasium и PPO-агент (Stable-Baselines3).
*   **[frontend/](file:///c:/Users/111/.gemini/antigravity-ide/scratch/babki_project/frontend)** — Одностраничное приложение (SPA) на **React + Vite**:
    *   Дизайн выполнен в минималистичном тёмном стиле **Strategy.com** (чистый чёрный фон, оранжевые биткоин-акценты, monospace-числа).
    *   Визуализация на **Recharts** и иконки **Lucide React**.

---

## 🚀 Инструкция по запуску

### 1. Запуск Backend (FastAPI)
Из корневой папки проекта активируйте виртуальное окружение и запустите сервер Uvicorn:
```bash
.\venv\Scripts\python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```
API будет доступно по адресу `http://127.0.0.1:8000`, интерактивная документация Swagger — `http://127.0.0.1:8000/docs`.

### 2. Запуск Frontend (React)
Перейдите в папку `frontend` и запустите Vite-сервер:
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
Интерфейс откроется в вашем браузере по адресу `http://127.0.0.1:5173/` (или `5174` при конфликте портов).
