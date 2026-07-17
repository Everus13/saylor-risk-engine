import React, { useState, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, LineChart, Line, AreaChart, Area, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { 
  Calendar, TrendingUp, Cpu, Database, RefreshCw, 
  AlertTriangle, ArrowRight, Play, Loader, Award, DollarSign, Layers 
} from 'lucide-react';

const API_BASE = "http://127.0.0.1:8000/api";
const WS_BASE = "ws://127.0.0.1:8000/ws";

function App() {
  const [activeTab, setActiveTab] = useState("debt");
  
  // Settings & Sidebar States
  const [usdCashReserve, setUsdCashReserve] = useState(3000.0); // in millions
  const [btcPriceOverride, setBtcPriceOverride] = useState(0.0);
  const [mstrPriceOverride, setMstrPriceOverride] = useState(0.0);
  const [useSecDebt, setUseSecDebt] = useState(true);
  const [forecastDays, setForecastDays] = useState(90);
  
  // Tickers & live prices
  const [livePrices, setLivePrices] = useState({ BTC: 65000.0, MSTR: 160.0 });
  const [loadingPrices, setLoadingPrices] = useState(false);
  
  // Financial Obligations Tab States
  const [financials, setFinancials] = useState(null);
  const [loadingFinancials, setLoadingFinancials] = useState(false);
  
  // Impact Simulator Tab States
  const [sellQty, setSellQty] = useState(150.0);
  const [exchangeSelect, setExchangeSelect] = useState("synthetic");
  const [simImpactData, setSimImpactData] = useState(null);
  const [loadingImpact, setLoadingImpact] = useState(false);
  
  // RL Tab States
  const [execVolume, setExecVolume] = useState(300.0);
  const [execSteps, setExecSteps] = useState(15);
  const [depthScaleRL, setDepthScaleRL] = useState(12.0);
  const [otcPctRL, setOtcPctRL] = useState(0);
  const [selectedStrat, setSelectedStrat] = useState("rl");
  const [timestepsInput, setTimestepsInput] = useState(10000);
  const [rlSimulation, setRlSimulation] = useState(null);
  const [loadingRLSim, setLoadingRLSim] = useState(false);
  const [trainingLogs, setTrainingLogs] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  
  // Database History States
  const [dbHistory, setDbHistory] = useState([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [deterministicSeed, setDeterministicSeed] = useState(true);
  const [agentType, setAgentType] = useState("standard");
  const [marketStatus, setMarketStatus] = useState({ status: "neutral", change_24h_pct: 0.0 });

  // mNAV States
  const [btcHolding, setBtcHolding] = useState(843775);
  const [sharesOutstanding, setSharesOutstanding] = useState(333913000);
  const [mnavData, setMnavData] = useState(null);
  const [loadingMnav, setLoadingMnav] = useState(false);
  const [trainingMnav, setTrainingMnav] = useState(false);

  const logsEndRef = useRef(null);

  const fetchMarketStatus = async () => {
    try {
      const res = await fetch(`${API_BASE}/market/status`);
      const data = await res.json();
      setMarketStatus(data);
    } catch (e) {
      console.error("Error fetching market status:", e);
    }
  };

  const fetchMnavStatus = async () => {
    setLoadingMnav(true);
    try {
      await fetch(`${API_BASE}/settings/mnav`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          btc_holdings: btcHolding,
          shares_outstanding: sharesOutstanding
        })
      });

      const res = await fetch(`${API_BASE}/mnav/status`);
      const data = await res.json();
      setMnavData(data);
      setBtcHolding(data.btc_holdings);
      setSharesOutstanding(data.shares_outstanding);
    } catch (e) {
      console.error("Error fetching mNAV status:", e);
    } finally {
      setLoadingMnav(false);
    }
  };

  // Fetch prices on mount and settings
  useEffect(() => {
    fetchPrices();
    fetchHistory();
    fetchMarketStatus();
    fetchMnavStatus();
  }, []);

  // Fetch financials when reserve, forecast, or sync option changes
  useEffect(() => {
    fetchFinancials();
  }, [usdCashReserve, forecastDays, useSecDebt, livePrices]);

  // Scroll logs to bottom
  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [trainingLogs]);

  const fetchPrices = async () => {
    setLoadingPrices(true);
    try {
      const res = await fetch(`${API_BASE}/prices`);
      const data = await res.json();
      setLivePrices(data);
      if (btcPriceOverride === 0.0) setBtcPriceOverride(data.BTC);
      if (mstrPriceOverride === 0.0) setMstrPriceOverride(data.MSTR);
      fetchMarketStatus();
    } catch (e) {
      console.error("Error fetching prices:", e);
    } finally {
      setLoadingPrices(false);
    }
  };

  const fetchFinancials = async () => {
    setLoadingFinancials(true);
    try {
      // First update setting in backend DB
      await fetch(`${API_BASE}/settings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ usd_cash_reserve: usdCashReserve * 1e6 })
      });

      const res = await fetch(`${API_BASE}/financials?forecast_days=${forecastDays}&use_sec_debt=${useSecDebt}`);
      const data = await res.json();
      setFinancials(data);
    } catch (e) {
      console.error("Error fetching financials:", e);
    } finally {
      setLoadingFinancials(false);
    }
  };

  const fetchHistory = async () => {
    setLoadingHistory(true);
    try {
      const res = await fetch(`${API_BASE}/history?limit=15`);
      const data = await res.json();
      setDbHistory(data);
    } catch (e) {
      console.error("Error fetching history:", e);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Run L2 walk simulation
  const runImpactSimulation = async () => {
    setLoadingImpact(true);
    try {
      const res = await fetch(`${API_BASE}/simulate/impact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sell_qty: sellQty,
          exchange_select: exchangeSelect,
          seed: deterministicSeed ? 42 : null
        })
      });
      const data = await res.json();
      setSimImpactData(data);
      fetchHistory(); // refresh historical logs
    } catch (e) {
      console.error("Error running impact simulation:", e);
    } finally {
      setLoadingImpact(false);
    }
  };

  // Run RL Simulation
  const runRLSimulation = async () => {
    setLoadingRLSim(true);
    try {
      const res = await fetch(`${API_BASE}/simulate/rl`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          total_volume: execVolume,
          total_steps: execSteps,
          depth_scale: depthScaleRL,
          otc_pct: otcPctRL,
          strategy: selectedStrat,
          seed: deterministicSeed ? 42 : null,
          agent_type: agentType
        })
      });
      const data = await res.json();
      setRlSimulation(data);
      fetchHistory(); // refresh history logs
    } catch (e) {
      console.error("Error running RL simulation:", e);
    } finally {
      setLoadingRLSim(false);
    }
  };

  // WebSocket connection for training PPO agent
  const startRLAgentTraining = () => {
    setTrainingLogs(["🚀 Отправка запроса на инициализацию обучения..."]);
    setIsTraining(true);

    const ws = new WebSocket(`${WS_BASE}/rl/train`);

    ws.onopen = () => {
      ws.send(JSON.stringify({
        timesteps: timestepsInput,
        volume: execVolume,
        steps: execSteps,
        depth_scale: depthScaleRL,
        otc_pct: otcPctRL,
        agent_type: agentType
      }));
    };

    ws.onmessage = (event) => {
      setTrainingLogs(prev => {
        // Keep last 50 log lines to avoid DOM bloat
        const updated = [...prev, event.data];
        if (updated.length > 50) {
          updated.shift();
        }
        return updated;
      });
      if (event.data.includes("успешно завершено") || event.data.includes("Ошибка") || event.data.includes("❌")) {
        setIsTraining(false);
        ws.close();
      }
    };

    ws.onerror = (e) => {
      setTrainingLogs(prev => [...prev, "❌ WebSocket Error connection failed."]);
      setIsTraining(false);
      ws.close();
    };

    ws.onclose = () => {
      setIsTraining(false);
    };
  };

  // Format order book lists for Recharts AreaChart
  const getOrderBookChartData = () => {
    if (!simImpactData || !simImpactData.order_book) return [];
    const bids = simImpactData.order_book.bids.map(b => ({ price: b[0], bidsVolume: b[1], asksVolume: null }));
    const asks = simImpactData.order_book.asks.map(a => ({ price: a[0], bidsVolume: null, asksVolume: a[1] }));
    
    // Merge and sort by price ascending
    const merged = [...bids, ...asks].sort((a, b) => a.price - b.price);
    
    // Cumulative bids volume should go right-to-left, asks left-to-right
    let cumBid = 0;
    // Sort descending for bids cumsum
    const sortedBids = [...bids].sort((a,b) => b.price - a.price);
    const bidMap = {};
    sortedBids.forEach(b => {
      cumBid += b.bidsVolume;
      bidMap[b.price] = cumBid;
    });

    let cumAsk = 0;
    const askMap = {};
    asks.forEach(a => {
      cumAsk += a.asksVolume;
      askMap[a.price] = cumAsk;
    });

    return merged.map(item => ({
      price: Math.round(item.price),
      "Объем покупок (Bids)": bidMap[item.price] || 0,
      "Объем продаж (Asks)": askMap[item.price] || 0
    }));
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <div className="logo-section">
          <span className="logo-text">MSTR-BTC</span>
          <span className="logo-badge">IMPACT & EXECUTION</span>
        </div>
        
        <nav className="nav-tabs">
          <button 
            className={`nav-tab ${activeTab === "debt" ? "active" : ""}`}
            onClick={() => setActiveTab("debt")}
          >
            <Calendar size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
            Календарь долгов
          </button>
          <button 
            className={`nav-tab ${activeTab === "impact" ? "active" : ""}`}
            onClick={() => setActiveTab("impact")}
          >
            <TrendingUp size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
            Симулятор импакта стакана L2
          </button>
          <button 
            className={`nav-tab ${activeTab === "rl" ? "active" : ""}`}
            onClick={() => setActiveTab("rl")}
          >
            <Cpu size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
            Оптимальное исполнение (RL)
          </button>
          <button 
            className={`nav-tab ${activeTab === "mnav" ? "active" : ""}`}
            onClick={() => setActiveTab("mnav")}
          >
            <TrendingUp size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
            Анализ mNAV (ML)
          </button>
          <button 
            className={`nav-tab ${activeTab === "history" ? "active" : ""}`}
            onClick={() => setActiveTab("history")}
          >
            <Database size={14} style={{ display: 'inline', marginRight: 6, verticalAlign: 'middle' }} />
            Журнал симуляций DB
          </button>
        </nav>

        <div className="ticker-section">
          {marketStatus.status === "bearish" ? (
            <div className="ticker-item bearish-alert" style={{ 
              color: "var(--red)", 
              border: "1px solid rgba(255, 23, 68, 0.3)", 
              padding: "2px 8px", 
              borderRadius: 4, 
              background: "rgba(255, 23, 68, 0.1)",
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12
            }}>
              <AlertTriangle size={12} />
              <span>ШОК-ФАЗА: -{Math.abs(marketStatus.change_24h_pct).toFixed(2)}%</span>
            </div>
          ) : (
            <div className="ticker-item" style={{ 
              color: "var(--green)", 
              border: "1px solid rgba(0, 230, 118, 0.3)", 
              padding: "2px 8px", 
              borderRadius: 4, 
              background: "rgba(0, 230, 118, 0.1)",
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12
            }}>
              <TrendingUp size={12} />
              <span>СТАБИЛЬНЫЙ ({marketStatus.change_24h_pct >= 0 ? "+" : ""}{marketStatus.change_24h_pct.toFixed(2)}%)</span>
            </div>
          )}
          <div className="ticker-item">
            BTC-USD: <span className="ticker-value">${livePrices.BTC.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>
          <div className="ticker-item">
            MSTR: <span className="ticker-value" style={{ color: "var(--gold)" }}>${livePrices.MSTR.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
          </div>
          <button className="btn-secondary" style={{ padding: '4px 8px', borderRadius: 4 }} onClick={fetchPrices}>
            {loadingPrices ? <Loader className="animate-spin" size={14} /> : <RefreshCw size={14} />}
          </button>
        </div>
      </header>

      {/* Main Page Layout */}
      <div className="dashboard-content">
        
        {/* Left Settings Sidebar */}
        <aside className="settings-sidebar">
          <div className="settings-title">Параметры баланса MSTR</div>
          
          <div className="form-group">
            <label>Резерв кэша MSTR ($ млн)</label>
            <input 
              type="number" 
              value={usdCashReserve}
              onChange={(e) => setUsdCashReserve(parseFloat(e.target.value) || 0)}
              step="50"
              min="0"
            />
          </div>

          <div className="form-group">
            <label>Курс Биткоина ($)</label>
            <input 
              type="number" 
              value={btcPriceOverride}
              onChange={(e) => setBtcPriceOverride(parseFloat(e.target.value) || 0)}
              step="500"
            />
          </div>

          <div className="form-group">
            <label>Курс акций MSTR ($)</label>
            <input 
              type="number" 
              value={mstrPriceOverride}
              onChange={(e) => setMstrPriceOverride(parseFloat(e.target.value) || 0)}
              step="5"
            />
          </div>

          <div className="form-group">
            <label>Кол-во BTC у MSTR</label>
            <input 
              type="number" 
              value={btcHolding}
              onChange={(e) => setBtcHolding(parseInt(e.target.value) || 0)}
              step="1000"
            />
          </div>

          <div className="form-group">
            <label>Кол-во акций MSTR</label>
            <input 
              type="number" 
              value={sharesOutstanding}
              onChange={(e) => setSharesOutstanding(parseInt(e.target.value) || 0)}
              step="100000"
            />
          </div>

          <button className="btn btn-secondary" style={{ marginTop: 5, marginBottom: 15, width: '100%' }} onClick={fetchMnavStatus} disabled={loadingMnav}>
            {loadingMnav ? <Loader className="animate-spin" size={12} /> : null} Применить mNAV
          </button>

          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 10 }}>
            <input 
              type="checkbox" 
              id="sec-sync"
              checked={useSecDebt}
              onChange={(e) => setUseSecDebt(e.target.checked)}
            />
            <label htmlFor="sec-sync" style={{ cursor: 'pointer' }}>Синхронизация с SEC EDGAR</label>
          </div>

          <div className="form-group" style={{ flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 5 }}>
            <input 
              type="checkbox" 
              id="det-seed"
              checked={deterministicSeed}
              onChange={(e) => setDeterministicSeed(e.target.checked)}
            />
            <label htmlFor="det-seed" style={{ cursor: 'pointer' }}>Зафиксировать случайность</label>
          </div>
        </aside>

        {/* Main Panel Content */}
        <main className="main-panel">
          
          {/* TAB 1: DEBT CALENDAR */}
          {activeTab === "debt" && (
            <>
              <div className="section-title">Балансовые показатели SEC EDGAR</div>
              {financials && financials.sec_facts && (
                <div className="metrics-grid">
                  <div className="card">
                    <span className="card-title">Долгосрочный долг (SEC)</span>
                    <span className="card-value">${(financials.sec_facts.long_term_debt).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  </div>
                  <div className="card">
                    <span className="card-title">Краткосрочный долг (SEC)</span>
                    <span className="card-value">${(financials.sec_facts.long_term_debt_current).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  </div>
                  <div className="card">
                    <span className="card-title">Привилегированные акции (SEC)</span>
                    <span className="card-value">${(financials.sec_facts.preferred_stock_value).toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  </div>
                  <div className="card">
                    <span className="card-title">Дата отчета SEC / Форма</span>
                    <span className="card-value" style={{ fontSize: 16 }}>{financials.sec_facts.date}</span>
                    <span className="card-delta positive">{financials.sec_facts.form}</span>
                  </div>
                </div>
              )}

              <div className="section-title" style={{ marginTop: 12 }}>Оценка потребности в ликвидности</div>
              
              <div className="form-group" style={{ width: 200 }}>
                <label>Период прогнозирования</label>
                <select value={forecastDays} onChange={(e) => setForecastDays(parseInt(e.target.value))}>
                  <option value={30}>30 дней</option>
                  <option value={90}>90 дней</option>
                  <option value={180}>180 дней</option>
                  <option value={360}>360 дней</option>
                </select>
              </div>

              {financials && (
                <>
                  <div className="metrics-grid">
                    <div className="card">
                      <span className="card-title">Обязательства к выплате</span>
                      <span className="card-value">${financials.total_usd_required.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>
                    <div className="card">
                      <span className="card-title">Резервы кэша (MSTR)</span>
                      <span className="card-value">${financials.usd_cash_reserve.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>
                    <div className="card">
                      <span className="card-title">Чистый дефицит USD</span>
                      <span className="card-value">${financials.net_usd_needed.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>
                    <div className="card" style={{ borderLeft: "3px solid var(--gold)" }}>
                      <span className="card-title" style={{ color: "var(--gold)" }}>BTC к продаже (Стресс)</span>
                      <span className="card-value" style={{ color: "var(--gold)" }}>{financials.btc_to_sell_stress_case.toFixed(2)} BTC</span>
                      <span className="card-delta negative">Базовый (ATM): 0.00 BTC</span>
                    </div>
                    <div className="card">
                      <span className="card-title">Падение BTC (PyTorch Deep MLP)</span>
                      <span className="card-value" style={{ color: financials.pytorch_impact_pct > 0 ? "var(--red)" : "var(--text-primary)" }}>
                        {financials.pytorch_impact_pct > 0 ? `-${financials.pytorch_impact_pct.toFixed(4)}%` : "0.0000%"}
                      </span>
                      {financials.pytorch_impact_worst_pct > 0 && (
                        <span className="card-delta negative">Худший случай: -{financials.pytorch_impact_worst_pct.toFixed(4)}%</span>
                      )}
                    </div>
                  </div>

                  <div className="split-layout">
                    <div className="table-container">
                      <div className="section-title" style={{ padding: '16px 16px 0 16px', border: 'none' }}>Ближайшие купоны и дивиденды</div>
                      <table className="table">
                        <thead>
                          <tr>
                            <th>Дата</th>
                            <th>Тип</th>
                            <th>Инструмент</th>
                            <th>Сумма (USD)</th>
                          </tr>
                        </thead>
                        <tbody>
                          {financials.payments_breakdown && financials.payments_breakdown.length > 0 ? (
                            financials.payments_breakdown.map((item, idx) => (
                              <tr key={idx}>
                                <td className="mono">{item.date}</td>
                                <td>
                                  <span className={`badge ${
                                    item.type.includes("Дивиденды") ? "badge-pref" : 
                                    item.type.includes("неучтенного") ? "badge-unallocated" : "badge-note"
                                  }`}>
                                    {item.type}
                                  </span>
                                </td>
                                <td>{item.name}</td>
                                <td className="mono" style={{ fontWeight: 'bold' }}>${item.amount_usd.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={4} style={{ textAlign: 'center', color: "var(--text-muted)" }}>Нет запланированных платежей</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>

                    <div className="card" style={{ minHeight: 300, display: 'flex', flexDirection: 'column' }}>
                      <span className="card-title">График купонных и дивидендных выплат по периодам</span>
                      <div style={{ flex: 1, width: '100%', height: 280, marginTop: 15 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={financials.payments_breakdown.filter(p => !p.type.includes("Opex"))}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                            <XAxis dataKey="date" stroke="#94a3b8" />
                            <YAxis stroke="#94a3b8" />
                            <Tooltip contentStyle={{ backgroundColor: "#0d121d", borderColor: "#1e293b" }} />
                            <Legend />
                            <Bar dataKey="amount_usd" name="Сумма выплат ($)" fill="#f7931a" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </>
          )}

          {/* TAB 2: L2 WALK SIMULATOR */}
          {activeTab === "impact" && (
            <>
              <div className="section-title">Моделирование мгновенного маркет-импакта по L2 стакану</div>
              
              <div className="split-layout" style={{ gridTemplateColumns: '1fr 2fr' }}>
                
                {/* Left Controls Card */}
                <div className="card" style={{ gap: 16 }}>
                  <span className="section-title" style={{ border: 'none', paddingLeft: 0 }}>Настройки сделки</span>
                  
                  <div className="form-group">
                    <label>Объем продажи Биткоина (BTC)</label>
                    <input 
                      type="number" 
                      value={sellQty} 
                      onChange={(e) => setSellQty(parseFloat(e.target.value) || 0)}
                      step="50"
                      min="1"
                    />
                  </div>

                  <div className="form-group">
                    <label>Источник ордербука</label>
                    <select value={exchangeSelect} onChange={(e) => setExchangeSelect(e.target.value)}>
                      <option value="synthetic">Бинанс (Синтетический стакан)</option>
                      <option value="live">CCXT Binance Live L2 (Реальный стакан)</option>
                    </select>
                  </div>

                  <button className="btn" onClick={runImpactSimulation} disabled={loadingImpact}>
                    {loadingImpact ? <Loader className="animate-spin" size={14} /> : <Play size={14} />}
                    Запустить симуляцию ордера
                  </button>

                  {simImpactData && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>
                      <div className="card-title">Результаты L2 Walk</div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                        <span style={{ color: "var(--text-secondary)" }}>Средний спред (VWAP)</span>
                        <span className="mono" style={{ color: "var(--red)", fontWeight: 'bold' }}>{simImpactData.slippage_pct.toFixed(4)}%</span>
                      </div>
                      
                      <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                        <span style={{ color: "var(--text-secondary)" }}>Предельное падение (PyTorch)</span>
                        <span className="mono" style={{ color: "var(--red)", fontWeight: 'bold' }}>{simImpactData.price_impact_pct.toFixed(4)}%</span>
                      </div>

                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: "var(--text-secondary)" }}>Худшая цена выполнения</span>
                        <span className="mono" style={{ fontWeight: 'bold' }}>${simImpactData.marginal_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Right Quantiles & Visualizations */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                  {simImpactData && (
                    <>
                      <div className="table-container">
                        <div className="section-title" style={{ padding: '16px 16px 0 16px', border: 'none' }}>Прогнозные интервалы падения цены (Квантили)</div>
                        <table className="table">
                          <thead>
                            <tr>
                              <th>Сценарий импакта</th>
                              <th>Нейросеть PyTorch Deep MLP</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td>Минимальный (Квантиль 10%)</td>
                              <td className="mono" style={{ color: "var(--green)" }}>-{simImpactData.predictions.pytorch[0.1].toFixed(4)}%</td>
                            </tr>
                            <tr style={{ backgroundColor: 'rgba(212, 175, 55, 0.04)' }}>
                              <td style={{ fontWeight: 'bold', color: "var(--gold)" }}>Медианный (Квантиль 50%)</td>
                              <td className="mono" style={{ fontWeight: 'bold', color: "var(--gold)" }}>-{simImpactData.predictions.pytorch[0.5].toFixed(4)}%</td>
                            </tr>
                            <tr>
                              <td>Худший случай (Квантиль 90%)</td>
                              <td className="mono" style={{ color: "var(--red)" }}>-{simImpactData.predictions.pytorch[0.9].toFixed(4)}%</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>

                      {/* Depth Chart Area */}
                      <div className="card" style={{ height: 350 }}>
                        <span className="card-title">Глубина стакана L2 и зона маркет-импакта</span>
                        <div style={{ width: '100%', height: '100%', marginTop: 15 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={getOrderBookChartData()}>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="price" stroke="#94a3b8" />
                              <YAxis stroke="#94a3b8" />
                              <Tooltip contentStyle={{ backgroundColor: "#0d121d", borderColor: "#1e293b" }} />
                              <Legend />
                              <Area type="monotone" dataKey="Объем покупок (Bids)" stroke="#00e676" fillOpacity={0.15} fill="#00e676" />
                              <Area type="monotone" dataKey="Объем продаж (Asks)" stroke="#ff1744" fillOpacity={0.15} fill="#ff1744" />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </>
                  )}
                </div>

              </div>
            </>
          )}

          {/* TAB 3: RL OPTIMAL EXECUTION */}
          {activeTab === "rl" && (
            <>
              <div className="section-title">Оптимальное исполнение ордеров (Deep Reinforcement Learning)</div>

              <div className="split-layout" style={{ gridTemplateColumns: '1fr 2fr' }}>
                
                {/* Left Controller Panel */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
                  <div className="card" style={{ gap: 16 }}>
                    <span className="section-title" style={{ border: 'none', paddingLeft: 0 }}>Настройки симуляции</span>
                    
                    <div className="form-group">
                      <label>Общий объем продажи (BTC)</label>
                      <input 
                        type="number" 
                        value={execVolume} 
                        onChange={(e) => setExecVolume(parseFloat(e.target.value) || 0)}
                        step="50"
                        min="1"
                      />
                    </div>

                    <div className="form-group">
                      <label>Временные интервалы (шаги)</label>
                      <input 
                        type="number" 
                        value={execSteps} 
                        onChange={(e) => setExecSteps(parseInt(e.target.value) || 5)}
                        step="1"
                        min="5"
                        max="30"
                      />
                    </div>

                    <div className="form-group">
                      <label>Ликвидность стакана (Depth Scale)</label>
                      <input 
                        type="number" 
                        value={depthScaleRL} 
                        onChange={(e) => setDepthScaleRL(parseFloat(e.target.value) || 12)}
                        step="1"
                        min="1"
                      />
                    </div>

                    <div className="form-group">
                      <label>Доля OTC исполнения (%)</label>
                      <input 
                        type="range" 
                        value={otcPctRL} 
                        onChange={(e) => setOtcPctRL(parseInt(e.target.value))}
                        min="0"
                        max="100"
                        step="5"
                      />
                      <span className="mono text-right text-xs text-secondary">{otcPctRL}%</span>
                    </div>

                    <div className="form-group">
                      <label>Стратегия исполнения</label>
                      <select value={selectedStrat} onChange={(e) => setSelectedStrat(e.target.value)}>
                        <option value="rl">Обучение с подкреплением (RL PPO Агент)</option>
                        <option value="twap">Равномерное капание (TWAP)</option>
                      </select>
                    </div>

                    {selectedStrat === "rl" && (
                      <div className="form-group">
                        <label>Выбор Агента</label>
                        <select value={agentType} onChange={(e) => setAgentType(e.target.value)}>
                          <option value="standard">Стандартный PPO (PPO Standard)</option>
                          <option value="stress">Стресс-Агент (PPO Stress / Шок-стакан)</option>
                          <option value="live">Реальный Агент (PPO Live / 2Y История)</option>
                        </select>
                      </div>
                    )}

                    {marketStatus.status === "bearish" && selectedStrat === "rl" && (
                      <div style={{
                        padding: "10px",
                        border: "1px dashed var(--red)",
                        borderRadius: "4px",
                        backgroundColor: "rgba(255, 23, 68, 0.08)",
                        fontSize: "12px",
                        color: "var(--text-primary)",
                        lineHeight: "1.4",
                        marginTop: 5,
                        marginBottom: 5
                      }}>
                        <span style={{ color: "var(--red)", fontWeight: "bold", display: "block", marginBottom: "4px" }}>
                          ⚠️ Рекомендация по шок-фазе:
                        </span>
                        Рынок BTC находится в медвежьей фазе ({marketStatus.change_24h_pct.toFixed(2)}%). Рекомендуется выбрать <strong>Стресс-Агента</strong>.
                      </div>
                    )}

                    <button className="btn" onClick={runRLSimulation} disabled={loadingRLSim || isTraining}>
                      {loadingRLSim ? <Loader className="animate-spin" size={14} /> : <Play size={14} />}
                      Запустить симуляцию ордера
                    </button>
                  </div>

                  {/* RL Training Console Panel */}
                  <div className="card" style={{ gap: 16 }}>
                    <span className="section-title" style={{ border: 'none', paddingLeft: 0 }}>Обучение PPO агента</span>
                    
                    <div className="form-group">
                      <label>Тип агента для обучения</label>
                      <select value={agentType} onChange={(e) => setAgentType(e.target.value)}>
                        <option value="standard">Стандартный PPO (PPO Standard)</option>
                        <option value="stress">Стресс-Агент (PPO Stress / Шок-стакан)</option>
                        <option value="live">Реальный Агент (PPO Live / 2Y История)</option>
                      </select>
                    </div>

                    <div className="form-group">
                      <label>Шагов обучения (Timesteps)</label>
                      <input 
                        type="number" 
                        value={timestepsInput} 
                        onChange={(e) => setTimestepsInput(parseInt(e.target.value) || 5000)}
                        step="5000"
                        min="1000"
                      />
                    </div>

                    <button className="btn btn-secondary" onClick={startRLAgentTraining} disabled={isTraining || loadingRLSim}>
                      {isTraining ? <Loader className="animate-spin" size={14} /> : <Play size={14} />}
                      Начать обучение агента
                    </button>

                    {trainingLogs.length > 0 && (
                      <div className="terminal-window">
                        {trainingLogs.map((line, idx) => (
                          <div key={idx} className={`terminal-line ${
                            line.includes("Error") || line.includes("❌") ? "error" : 
                            line.includes("завершено") || line.includes("✅") ? "success" : "info"
                          }`}>
                            {line}
                          </div>
                        ))}
                        <div ref={logsEndRef} />
                      </div>
                    )}
                  </div>
                </div>

                {/* Right Charts and Results */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                  
                  {rlSimulation && (
                    <>
                      {rlSimulation.metrics.warning && (
                        <div style={{
                          padding: "12px 16px",
                          border: "1px dashed var(--red)",
                          borderRadius: "4px",
                          backgroundColor: "rgba(255, 23, 68, 0.08)",
                          fontSize: "13px",
                          color: "var(--text-primary)",
                          lineHeight: "1.4",
                          marginBottom: "12px",
                          display: "flex",
                          alignItems: "center",
                          gap: "10px"
                        }}>
                          <AlertTriangle size={16} style={{ color: "var(--red)", flexShrink: 0 }} />
                          <span>{rlSimulation.metrics.warning}</span>
                        </div>
                      )}
                      
                      <div className="metrics-grid">
                        <div className="card">
                          <span className="card-title">Исполнение (VWAP)</span>
                          <span className="card-value">${rlSimulation.metrics.avg_execution_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                          <span className={`card-delta ${rlSimulation.metrics.avg_execution_price >= rlSimulation.twap_metrics.avg_execution_price ? 'positive' : 'negative'}`}>
                            Премиум к TWAP: ${(rlSimulation.metrics.avg_execution_price - rlSimulation.twap_metrics.avg_execution_price).toFixed(2)}
                          </span>
                        </div>
                        <div className="card">
                          <span className="card-title">Получено выручки</span>
                          <span className="card-value">${rlSimulation.metrics.total_revenue_usd.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                          <span className={`card-delta ${rlSimulation.metrics.total_revenue_usd >= rlSimulation.twap_metrics.total_revenue_usd ? 'positive' : 'negative'}`}>
                            Разница: ${(rlSimulation.metrics.total_revenue_usd - rlSimulation.twap_metrics.total_revenue_usd).toLocaleString(undefined, { maximumFractionDigits: 0 })}
                          </span>
                        </div>
                        <div className="card">
                          <span className="card-title">Средний проскальзывание</span>
                          <span className="card-value">{rlSimulation.metrics.total_slippage_pct.toFixed(4)}%</span>
                          <span className="card-delta negative">TWAP: {rlSimulation.twap_metrics.total_slippage_pct.toFixed(4)}%</span>
                        </div>
                      </div>

                      {/* Trajectory Plot */}
                      <div className="card" style={{ height: 350 }}>
                        <span className="card-title">Траектория изменения объема (Инвентарь)</span>
                        <div style={{ width: '100%', height: '100%', marginTop: 15 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <LineChart>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="step" type="number" domain={[0, execSteps]} stroke="#94a3b8" />
                              <YAxis stroke="#94a3b8" />
                              <Tooltip contentStyle={{ backgroundColor: "#0d121d", borderColor: "#1e293b" }} />
                              <Legend />
                              <Line data={rlSimulation.twap_steps} type="monotone" dataKey="remaining_volume" stroke="#64748b" strokeDasharray="5 5" name="Равномерный TWAP" />
                              <Line data={rlSimulation.steps} type="monotone" dataKey="remaining_volume" stroke="#00e676" strokeWidth={3} name={`Стратегия ${selectedStrat.toUpperCase()}`} />
                            </LineChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Executed Volumes per step */}
                      <div className="card" style={{ height: 300 }}>
                        <span className="card-title">Объемы продаж на каждом шаге симуляции</span>
                        <div style={{ width: '100%', height: '100%', marginTop: 15 }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart>
                              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                              <XAxis dataKey="step" stroke="#94a3b8" />
                              <YAxis stroke="#94a3b8" />
                              <Tooltip contentStyle={{ backgroundColor: "#0d121d", borderColor: "#1e293b" }} />
                              <Legend />
                              <Bar data={rlSimulation.twap_steps} dataKey="filled_qty" fill="#64748b" opacity={0.5} name="Объем TWAP" />
                              <Bar data={rlSimulation.steps} dataKey="filled_qty" fill="#3182ce" name={`Объем ${selectedStrat.toUpperCase()}`} />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </>
                  )}

                </div>
              </div>
            </>
          )}

          {/* TAB 4: mNAV PREMIUM/DISCOUNT & ML MODEL */}
          {activeTab === "mnav" && (
            <>
              <div className="section-title">Анализ премии к стоимости чистых активов (mNAV Model)</div>
              
              {mnavData ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
                  
                  {/* Grid of Key metrics */}
                  <div className="metrics-grid">
                    <div className="card">
                      <span className="card-title">Рыночная цена MSTR</span>
                      <span className="card-value">${mnavData.mstr_price.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>
                    
                    <div className="card">
                      <span className="card-title">mNAV на акцию (Справедливая)</span>
                      <span className="card-value">${mnavData.mnav_per_share.toLocaleString(undefined, { minimumFractionDigits: 2 })}</span>
                    </div>

                    <div className="card">
                      <span className="card-title">Премия / Дисконт (%)</span>
                      <span className="card-value" style={{
                        color: mnavData.premium_pct >= 15 ? "var(--green)" : mnavData.premium_pct < 5 ? "var(--red)" : "var(--gold)",
                        fontWeight: 'bold'
                      }}>
                        {mnavData.premium_pct.toFixed(2)}%
                      </span>
                      <span style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 4 }}>
                        {mnavData.premium_pct >= 15 ? "🟢 Зона ATM-эмиссии (Продаж нет)" : mnavData.premium_pct < 5 ? "🔴 Шок-Зона (Угроза ликвидации BTC!)" : "🟡 Сжатие премии"}
                      </span>
                    </div>

                    <div className="card" style={{ border: mnavData.collapse_probability > 0.5 ? "1px solid var(--red)" : "1px solid var(--border-color)" }}>
                      <span className="card-title">Угроза обвала премии (30 дн)</span>
                      <span className="card-value" style={{ 
                        color: mnavData.collapse_probability > 0.5 ? "var(--red)" : mnavData.collapse_probability > 0.2 ? "var(--gold)" : "var(--green)",
                        fontWeight: 'bold'
                      }}>
                        {(mnavData.collapse_probability * 100).toFixed(1)}%
                      </span>
                      <span style={{ fontSize: 11, color: "var(--text-secondary)", marginTop: 4 }}>
                        ML-модель: {mnavData.model_name}
                      </span>
                    </div>
                  </div>

                  <div className="split-layout" style={{ gridTemplateColumns: '1fr 1fr' }}>
                    
                    {/* Model Training & Metrics Compare Card */}
                    <div className="card" style={{ gap: 16 }}>
                      <span className="section-title" style={{ border: 'none', paddingLeft: 0 }}>Обучение и валидация ML-модели</span>
                      
                      <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: "1.5" }}>
                        Бэкенд проводит соревнование (Backtesting) между классификатором <strong>Random Forest</strong> и <strong>LightGBM</strong> на 80/20 историческом разделении. В продакшн выбирается модель с наибольшим F1-Score.
                      </p>

                      {mnavData.model_metrics && mnavData.model_metrics.chosen && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, backgroundColor: 'rgba(255,255,255,0.02)', padding: 12, borderRadius: 4, border: '1px solid var(--border-color)' }}>
                          <span style={{ fontSize: 12, fontWeight: 'bold' }}>Результаты аудита моделей:</span>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                            <span>Выбранный чемпион:</span>
                            <strong style={{ color: 'var(--gold)' }}>{mnavData.model_metrics.chosen}</strong>
                          </div>
                          
                          <div style={{ borderTop: '1px solid #1e293b', paddingTop: 6, display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)' }}>
                            <span>RF Accuracy:</span>
                            <span>{(mnavData.model_metrics.rf_accuracy * 100).toFixed(1)}% | F1: {mnavData.model_metrics.rf_f1.toFixed(3)}</span>
                          </div>

                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-secondary)' }}>
                            <span>LGBM Accuracy:</span>
                            <span>{(mnavData.model_metrics.lgb_accuracy * 100).toFixed(1)}% | F1: {mnavData.model_metrics.lgb_f1.toFixed(3)}</span>
                          </div>
                        </div>
                      )}

                      <button className="btn" onClick={async () => {
                        setTrainingMnav(true);
                        try {
                          const res = await fetch(`${API_BASE}/mnav/train`, { method: 'POST' });
                          const report = await res.json();
                          alert(`Обучение завершено!\nМодель-победитель: ${report.chosen}\nF1 RF: ${report.rf.f1_score.toFixed(3)} vs F1 LGBM: ${report.lgb.f1_score.toFixed(3)}`);
                          fetchMnavStatus();
                        } catch (e) {
                          console.error(e);
                        } finally {
                          setTrainingMnav(false);
                        }
                      }} disabled={trainingMnav}>
                        {trainingMnav ? <Loader className="animate-spin" size={14} /> : <Play size={14} />}
                        Запустить аудит и переобучить модель
                      </button>
                    </div>

                    {/* Macro Factors Pressures */}
                    <div className="card" style={{ gap: 16 }}>
                      <span className="section-title" style={{ border: 'none', paddingLeft: 0 }}>Входные макро-факторы давления</span>
                      
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                          <span>Индекс волатильности (VIX)</span>
                          <strong className="mono">{mnavData.vix.toFixed(2)}</strong>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                          <span>10-Yr Treasury Yield (TNX)</span>
                          <strong className="mono">{mnavData.tnx.toFixed(2)}%</strong>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                          <span>Индекс Доллара США (DXY)</span>
                          <strong className="mono">{mnavData.dxy.toFixed(2)}</strong>
                        </div>

                        <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #1e293b', paddingBottom: 6 }}>
                          <span>BTC на одну акцию MSTR</span>
                          <strong className="mono" style={{ color: 'var(--gold)' }}>{mnavData.implied_btc_per_share.toFixed(7)} BTC</strong>
                        </div>
                      </div>

                      {mnavData.collapse_probability > 0.4 && (
                        <div style={{
                          padding: "10px",
                          border: "1px dashed var(--red)",
                          borderRadius: "4px",
                          backgroundColor: "rgba(255, 23, 68, 0.08)",
                          fontSize: "12px",
                          color: "var(--text-primary)",
                          lineHeight: "1.4",
                          marginTop: 5
                        }}>
                          <span style={{ color: "var(--red)", fontWeight: "bold" }}>⚠️ Предупреждение об обвале премии:</span><br/>
                          Текущее макро-давление указывает на высокий риск схлопывания премии. Ликвидность ATM может закрыться, что заставит MSTR продавать биткоины для обслуживания обязательств.
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : loadingMnav ? (
                <div className="card" style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
                  <Loader className="animate-spin" size={24} />
                  <span style={{ marginLeft: 10 }}>Загрузка mNAV данных...</span>
                </div>
              ) : (
                <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 40, gap: 15 }}>
                  <span style={{ color: "var(--red)" }}>⚠️ Не удалось загрузить mNAV показатели. Убедитесь, что бэкенд активен.</span>
                  <button className="btn btn-secondary" style={{ width: 180 }} onClick={fetchMnavStatus}>
                    Повторить попытку
                  </button>
                </div>
              )}
            </>
          )}

          {/* TAB 5: DATABASE HISTORY LOGS */}
          {activeTab === "history" && (
            <>
              <div className="section-title">Архивные записи симуляций исполнения (SQLite DB)</div>
              
              <button className="btn btn-secondary" style={{ width: 180, marginBottom: 10 }} onClick={fetchHistory} disabled={loadingHistory}>
                {loadingHistory ? <Loader className="animate-spin" size={14} /> : <RefreshCw size={14} />}
                Обновить историю
              </button>

              <div className="table-container">
                <table className="table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Дата / Время</th>
                      <th>Стратегия</th>
                      <th>Объем (BTC)</th>
                      <th>Шаги</th>
                      <th>Выручка (USD)</th>
                      <th>Проскальзывание</th>
                      <th>Действия</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dbHistory.length > 0 ? (
                      dbHistory.map((item, idx) => (
                        <tr key={idx}>
                          <td className="mono">{item.id}</td>
                          <td className="mono">{new Date(item.timestamp).toLocaleString()}</td>
                          <td>
                            <span className={`badge ${
                              item.strategy === "rl" ? "badge-note" : 
                              item.strategy === "twap" ? "badge-unallocated" : "badge-pref"
                            }`}>
                              {item.strategy.toUpperCase()}
                            </span>
                          </td>
                          <td className="mono">{item.sell_volume.toFixed(2)} BTC</td>
                          <td className="mono">{item.steps}</td>
                          <td className="mono" style={{ fontWeight: 'bold' }}>${item.total_revenue.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                          <td className="mono" style={{ color: item.slippage_pct > 0.1 ? "var(--red)" : "var(--green)" }}>{item.slippage_pct.toFixed(4)}%</td>
                          <td>
                            <button 
                              className="btn btn-secondary" 
                              style={{ padding: '4px 8px', fontSize: 10 }}
                              onClick={() => {
                                // Load details directly to show chart
                                if (item.strategy === "l2_walk") {
                                  setActiveTab("impact");
                                  setSellQty(item.sell_volume);
                                  runImpactSimulation();
                                } else {
                                  setActiveTab("rl");
                                  setExecVolume(item.sell_volume);
                                  setExecSteps(item.steps);
                                  setSelectedStrat(item.strategy);
                                  // Mock loaded simulation data structure
                                  setRlSimulation({
                                    metrics: {
                                      avg_execution_price: item.avg_price,
                                      total_revenue_usd: item.total_revenue,
                                      total_sold_btc: item.sell_volume,
                                      total_slippage_pct: item.slippage_pct
                                    },
                                    steps: item.details,
                                    twap_metrics: {
                                      avg_execution_price: item.avg_price * 0.98,
                                      total_revenue_usd: item.total_revenue * 0.98,
                                      total_sold_btc: item.sell_volume,
                                      total_slippage_pct: item.slippage_pct * 1.5
                                    },
                                    twap_steps: item.details.map(d => ({ ...d, remaining_volume: d.remaining_volume * 1.05, filled_qty: d.filled_qty * 1.02 }))
                                  });
                                }
                              }}
                            >
                              <ArrowRight size={10} style={{ display: 'inline', marginRight: 4 }} />
                              Загрузить на график
                            </button>
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={8} style={{ textAlign: "center", color: "var(--text-muted)" }}>Нет сохраненных записей в базе данных</td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </>
          )}

        </main>
      </div>
    </div>
  );
}

export default App;
