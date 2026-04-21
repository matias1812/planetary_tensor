"""
MOTOR DE ANÁLISIS PLANETARIO UNIVERSAL
========================================
Pipeline completo: Efemérides → VIX → GDELT → Correlación → Features

Dependencias:
    pip install skyfield yfinance pandas numpy scipy requests tqdm matplotlib seaborn

Datos:
    - Efemérides: skyfield (NASA JPL de406.bsp — se descarga automáticamente)
    - VIX:        yfinance (Yahoo Finance, desde 1990)
    - GDELT:      API pública v2 (sin autenticación)

Uso:
    python planetary_tensor_analysis.py
    python planetary_tensor_analysis.py --start 1990-01-01 --end 2024-01-01
    python planetary_tensor_analysis.py --date 2024-10-15  # análisis de fecha única
"""

import argparse
import time
import warnings
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import RidgeCV, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ─────────────────────────────────────────────

PLANETS = {
    "sun":     {"name": "Sol",      "layer": "fast",  "weight": 1.0, "archetype": "identidad/vitalidad"},
    "moon":    {"name": "Luna",     "layer": "fast",  "weight": 1.2, "archetype": "emoción/ciclos"},
    "mercury": {"name": "Mercurio", "layer": "fast",  "weight": 0.8, "archetype": "comunicación/redes"},
    "venus":   {"name": "Venus",    "layer": "fast",  "weight": 0.7, "archetype": "valor/cohesión"},
    "mars":    {"name": "Marte",    "layer": "fast",  "weight": 1.1, "archetype": "acción/conflicto"},
    "jupiter": {"name": "Júpiter",  "layer": "slow",  "weight": 1.3, "archetype": "expansión/ley"},
    "saturn":  {"name": "Saturno",  "layer": "slow",  "weight": 1.5, "archetype": "estructura/límite"},
    "uranus":  {"name": "Urano",    "layer": "trans", "weight": 1.4, "archetype": "disrupción/tecnología"},
    "neptune": {"name": "Neptuno",  "layer": "trans", "weight": 1.2, "archetype": "ilusión/disolución"},
    "pluto":   {"name": "Plutón",   "layer": "trans", "weight": 1.6, "archetype": "transformación/poder"},
}

ASPECTS = {
    "conjunction": {"angle": 0,   "tension": 0.7, "orb": 8},
    "sextile":     {"angle": 60,  "tension": 0.3, "orb": 4},
    "square":      {"angle": 90,  "tension": 1.0, "orb": 8},
    "trine":       {"angle": 120, "tension": 0.2, "orb": 6},
    "quincunx":    {"angle": 150, "tension": 0.6, "orb": 3},
    "opposition":  {"angle": 180, "tension": 0.9, "orb": 8},
}

VIX_CRITICAL = 35.0   # umbral de pánico
WINDOW_DAYS  = 14     # ventana de correlación (±días)
CORR_THRESH  = 0.35   # |r| mínimo para selección de features


# ─────────────────────────────────────────────
# 2. EFEMÉRIDES (skyfield)
# ─────────────────────────────────────────────

def load_ephemeris():
    """Descarga el kernel JPL si no existe y carga las efemérides."""
    from skyfield.api import Loader
    load = Loader("./skyfield_data")
    eph  = load("de406.bsp")   # cubre 1850-2150
    ts   = load.timescale()
    return eph, ts


def get_ecliptic_lon(body, ts_time, eph):
    """Longitud eclíptica geocéntrica (0-360°) de un cuerpo en un instante dado."""
    from skyfield.api import wgs84
    from skyfield import framelib
    earth = eph["earth"]
    astrometric = earth.at(ts_time).observe(body)
    lat, lon, _ = astrometric.ecliptic_latlon(epoch="date")
    return lon.degrees % 360


def compute_daily_positions(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Calcula las posiciones eclípticas vectorizadas (velocidad extrema).
    Usa numpy vectorizado de Skyfield en vez de un bucle día-a-día.
    """
    print("\n[1/4] Calculando posiciones planetarias (Vectorizado)...")
    eph, ts = load_ephemeris()
    earth = eph["earth"]

    planet_names = ["sun", "moon", "mercury barycenter", "venus barycenter",
                    "mars barycenter", "jupiter barycenter", "saturn barycenter",
                    "uranus barycenter", "neptune barycenter", "pluto barycenter"]
    keys = ["sun", "moon", "mercury", "venus", "mars",
            "jupiter", "saturn", "uranus", "neptune", "pluto"]

    dates = pd.date_range(start_date, end_date, freq="D")

    # Vectorización: pasar todo el arreglo de tiempo de una sola vez
    # .to_numpy() es necesario porque jplephem muta internamente los arrays
    t = ts.utc(dates.year.to_numpy(), dates.month.to_numpy(), dates.day.to_numpy())

    data = {"date": dates}
    for key, name in zip(keys, planet_names):
        body = eph[name]
        astrometric = earth.at(t).observe(body)
        lat, lon, _ = astrometric.ecliptic_latlon(epoch="date")
        data[key] = lon.degrees % 360

    df = pd.DataFrame(data).set_index("date")
    return df


# ─────────────────────────────────────────────
# 3. CÁLCULO DE ASPECTOS Y TENSOR DE TENSIÓN
# ─────────────────────────────────────────────

def angular_distance(lon1: float, lon2: float) -> float:
    """Distancia angular mínima entre dos longitudes eclípticas (0-180°)."""
    diff = abs(lon1 - lon2) % 360
    return diff if diff <= 180 else 360 - diff


def compute_aspect_tension(lon1: float, lon2: float,
                           w1: float, w2: float) -> dict:
    """
    Evalúa todos los aspectos entre dos cuerpos.

    Returns:
        dict con la tensión total y los aspectos activos.
    """
    angle  = angular_distance(lon1, lon2)
    total  = 0.0
    active = []

    for asp_name, asp in ASPECTS.items():
        delta = abs(angle - asp["angle"])
        if delta <= asp["orb"]:
            exactness = 1.0 - delta / asp["orb"]
            tension   = asp["tension"] * exactness * w1 * w2
            total    += tension
            active.append({
                "aspect":    asp_name,
                "tension":   round(tension, 4),
                "exactness": round(exactness, 3),
                "orb_real":  round(delta, 2),
            })

    return {"total_tension": total, "active": active}


def build_tensor(positions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el tensor diario de tensión para los 66 pares planetarios.
    [OPTIMIZADO]: Usa operaciones matriciales NumPy en vez de iterrows.
    """
    print("\n[2/4] Construyendo tensor de 66 pares (Cálculo Matricial)...")
    planet_keys = list(PLANETS.keys())
    pairs = [(planet_keys[i], planet_keys[j])
             for i in range(len(planet_keys))
             for j in range(i+1, len(planet_keys))]

    tensor_df = pd.DataFrame(index=positions_df.index)
    global_tension = np.zeros(len(positions_df))
    max_possible = len(pairs) * 1.6 * 2.0

    for p1, p2 in tqdm(pairs, desc="Pares Planetarios"):
        w1 = PLANETS[p1]["weight"]
        w2 = PLANETS[p2]["weight"]

        lon1 = positions_df[p1].values
        lon2 = positions_df[p2].values

        # Distancia angular vectorizada (0-180°)
        diff = np.abs(lon1 - lon2) % 360
        angle = np.where(diff <= 180, diff, 360 - diff)

        pair_tension = np.zeros(len(positions_df))

        for asp_name, asp in ASPECTS.items():
            delta = np.abs(angle - asp["angle"])
            mask = delta <= asp["orb"]
            if np.any(mask):
                exactness = 1.0 - (delta[mask] / asp["orb"])
                tension = asp["tension"] * exactness * w1 * w2
                pair_tension[mask] += tension

        col = f"{p1}_{p2}_tension"
        tensor_df[col] = np.round(pair_tension, 4)
        global_tension += pair_tension

    tensor_df["global_tension"] = np.round(
        np.clip(global_tension / max_possible * 100, 0, 100), 2
    )
    return tensor_df


# ─────────────────────────────────────────────
# 4. SEÑALES EXTERNAS
# ─────────────────────────────────────────────

def fetch_vix(start_date: str, end_date: str) -> pd.Series:
    """Descarga el índice VIX desde Yahoo Finance."""
    print("\n[3/4] Descargando VIX...")
    import yfinance as yf
    vix = yf.download("^VIX", start=start_date, end=end_date,
                      progress=False, auto_adjust=False)["Close"].squeeze()
    vix.index = pd.to_datetime(vix.index).tz_localize(None)
    vix.name  = "vix"
    # Rellenar fines de semana y festivos
    vix = vix.reindex(pd.date_range(start_date, end_date, freq="D")).ffill()
    return vix


def fetch_gdelt_conflict(start_date: str, end_date: str,
                         query: str = "cyberattack OR protest OR strike OR riot") -> pd.Series:
    """
    Descarga volumen de artículos en GDELT v2.
    [OPTIMIZADO]: Caché local + peticiones concurrentes con ThreadPoolExecutor.
    """
    print("[3/4] Descargando datos GDELT (Modo Rápido)...")
    cache_dir = Path("./output/.gdelt_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    base = "https://api.gdeltproject.org/api/v2/doc/doc"
    dates = pd.date_range(start_date, end_date, freq="W")
    target_dates = dates[:52]
    counts = {}
    dates_to_fetch = []
    
    # Leer caché
    for d in target_dates:
        cache_file = cache_dir / f"{d.strftime('%Y%m%d')}.json"
        if cache_file.exists():
            counts[d] = json.loads(cache_file.read_text())["count"]
        else:
            dates_to_fetch.append(d)
    
    if dates_to_fetch:
        def _fetch_one(d):
            retries = 2
            while retries > 0:
                try:
                    params = {
                        "query": query, "mode": "artlist", "maxrecords": 250,
                        "startdatetime": d.strftime("%Y%m%d000000"),
                        "enddatetime": (d + timedelta(days=6)).strftime("%Y%m%d235959"),
                        "format": "json",
                    }
                    resp = requests.get(base, params=params, timeout=10)
                    if resp.status_code == 200:
                        n = len(resp.json().get("articles", []))
                        cache_file = cache_dir / f"{d.strftime('%Y%m%d')}.json"
                        cache_file.write_text(json.dumps({"count": n}))
                        return d, n
                    elif resp.status_code == 429:
                        time.sleep(1)
                        retries -= 1
                    else:
                        retries -= 1
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    retries -= 1
            return d, np.nan
        
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = {pool.submit(_fetch_one, d): d for d in dates_to_fetch}
            for f in tqdm(as_completed(futures), total=len(dates_to_fetch), desc="Semanas GDELT"):
                d, n = f.result()
                counts[d] = n

    s = pd.Series(counts).reindex(
        pd.date_range(start_date, end_date, freq="D")
    ).interpolate().ffill().bfill()

    # Z-Score Móvil de 90 días (aborda la no-estacionalidad del volumen web histórico)
    s_mean = s.rolling(window=90, min_periods=30).mean()
    s_std = s.rolling(window=90, min_periods=30).std().replace(0, np.nan)
    s_zscore = (s - s_mean) / s_std
    
    s = s_zscore.fillna(0).clip(lower=-3, upper=5)
    s.name = "gdelt_conflict"
    return s


def fetch_commodities(start_date: str, end_date: str) -> pd.DataFrame:
    """Descarga precios de Oro y Cobre (Capa Física Fuerte)."""
    print("[3/4] Descargando Commodities (Oro, Cobre)...")
    import yfinance as yf
    tickers = {"GC=F": "gold", "HG=F": "copper"}
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, progress=False, auto_adjust=False)
    
    # Manejar MultiIndex (Close)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
        
    closes = closes.rename(columns=tickers)
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.reindex(pd.date_range(start_date, end_date, freq="D")).ffill().bfill()
    return closes


# ─────────────────────────────────────────────
# 5. CORRELACIÓN Y SELECCIÓN DE FEATURES
# ─────────────────────────────────────────────

def build_ml_dataset(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, 
                     max_lag: int = 30) -> dict:
    """
    Construye el dataset para Machine Learning.
    - Autómatas: Rezagos del VIX (vix_lag_1 a vix_lag_5).
    - Features planetarios: Rezagos de todas las tensiones (k=1 a max_lag).
    - Target: VIX_t y VIX_extreme_t (VIX > 30).
    - Split: Train (hasta 2018), Test (desde 2019).
    """
    print("\n[4/4] Construyendo Features (Rezagos y Preparación ML)...")
    
    # 1. Alineación base (ignorar gdelt para ML por su límite de 1 año)
    df = tensor_df.join(signals_df[["vix"]]).dropna()
    
    # 2. Rezagos del Baseline (AR5 del VIX + Ancla de Hoy)
    baseline_cols = ["vix"]
    for k in range(1, 6):
        col_name = f"vix_lag_{k}"
        df[col_name] = df["vix"].shift(k)
        baseline_cols.append(col_name)
        
    # 3. Features planetarios
    pair_cols = [c for c in tensor_df.columns if c.endswith("_tension")]
    planet_lags = []
    fast_keywords = ["sun_", "_sun", "moon_", "_moon", "mercury_", "_mercury"]
    
    for col in pair_cols:
        is_fast = any(k in col for k in fast_keywords)
        if is_fast:
            # Lags para pares rápidos (alta frecuencia) para captar ondas cortas
            for k in range(1, max_lag + 1, 3):  
                lag_name = f"{col}_lag_{k}"
                df[lag_name] = df[col].shift(k)
                planet_lags.append(lag_name)
        else:
            # Sin lags diarios para pares lentos (gravedad macro), evitar colinealidad
            df[col] = tensor_df[col]
            planet_lags.append(col)
            
    # 4. Target variables (Caza de Cisnes Negros: Solo ruptura inicial)
    df["target_vix"] = df["vix"].shift(-1) - df["vix"] # Predice Velocidad, no posición (por ML Ridge)
    df["target_extreme"] = ((df["vix"].shift(-1) > 30) & (df["vix"] < 25)).astype(int)
    
    # Drop NAs causados por el shift
    df = df.dropna()
    
    # 5. Train/Test Split (Dinámico: 80% Train / 20% Test secuencial)
    split_idx = int(len(df) * 0.8)
    split_date = df.index[split_idx]
    
    print(f"   [ML Split] Entrenando hasta: {split_date.date()} | Testeando desde: {split_date.date()}")
    
    train_mask = df.index < split_date
    test_mask  = df.index >= split_date
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("ADVERTENCIA: Rango insuficiente para split. Usando todo como Train.")
        train_mask = df.index.notnull()
        test_mask = df.index.notnull()
        
    X_baseline = df[baseline_cols]
    X_planets  = df[baseline_cols + planet_lags]
    y_clf      = df["target_extreme"]
    
    return {
        "X_base_train": X_baseline[train_mask], "X_base_test": X_baseline[test_mask],
        "X_plan_train": X_planets[train_mask],  "X_plan_test": X_planets[test_mask],
        "y_clf_train":  y_clf[train_mask],      "y_clf_test":  y_clf[test_mask],
        "features": planet_lags,
        "test_dates": df[test_mask].index
    }

def train_evaluate_models(dataset: dict, out_dir: Path):
    """
    Entrena modelos de Regularización y Baselines, evaluando Out-Of-Sample.
    """
    print("\n=== EVALUACIÓN DE MODELOS (OUT-OF-SAMPLE) ===")
    
    # Normalización
    scaler_b = StandardScaler().fit(dataset["X_base_train"])
    scaler_p = StandardScaler().fit(dataset["X_plan_train"])
    
    Xbt_scaled, Xbte_scaled = scaler_b.transform(dataset["X_base_train"]), scaler_b.transform(dataset["X_base_test"])
    Xpt_scaled, Xpte_scaled = scaler_p.transform(dataset["X_plan_train"]), scaler_p.transform(dataset["X_plan_test"])
    
    # ---------------------------------------------------------
    # CLASIFICACIÓN: Eventos Extremos (VIX > 30) (Tail-Risk)
    # ---------------------------------------------------------
    print("\n>> Entrenando Clasificación de Eventos Extremos (VIX > 30)...")
    if sum(dataset["y_clf_train"]) < 5 or sum(dataset["y_clf_test"]) < 5:
        print("   Incluso con 30 días, pocos datos de estrés para clasificador realista. Saltando métricas.")
        metrics = {"Planetary_R2": 0, "Baseline_R2": 0}
        plan_r2, base_r2 = 0, 0
    else:
        # Baseline
        base_clf = LogisticRegressionCV(max_iter=1000)
        base_clf.fit(Xbt_scaled, dataset["y_clf_train"])
        b_preds = base_clf.predict(Xbte_scaled)
        
        # Planetario (Ridge/L2)
        plan_clf = LogisticRegressionCV(max_iter=1000, penalty='l2')
        plan_clf.fit(Xpt_scaled, dataset["y_clf_train"])
        p_preds = plan_clf.predict(Xpte_scaled)
        
        bp = precision_score(dataset["y_clf_test"], b_preds, zero_division=0)
        br = recall_score(dataset["y_clf_test"], b_preds, zero_division=0)
        ba = roc_auc_score(dataset["y_clf_test"], base_clf.predict_proba(Xbte_scaled)[:, 1])
        
        pp = precision_score(dataset["y_clf_test"], p_preds, zero_division=0)
        pr = recall_score(dataset["y_clf_test"], p_preds, zero_division=0)
        pa = roc_auc_score(dataset["y_clf_test"], plan_clf.predict_proba(Xpte_scaled)[:, 1])
        
        print(f"   Baseline AR5    -> AUC: {ba:.3f} | Precision: {bp:.3f} | Recall: {br:.3f}")
        print(f"   Planetary Ridge -> AUC: {pa:.3f} | Precision: {pp:.3f} | Recall: {pr:.3f}")
        
        metrics = {
            "Planetary_AUC": pa, "Baseline_AUC": ba
        }
        
    return metrics, dataset["test_dates"]


# ─────────────────────────────────────────────
# 5b. MULTIPLICADOR DE CASCADA
# ─────────────────────────────────────────────

# Features estructurales lentos que sobrevivieron al Ridge L2 (hardcoded del análisis de 36 años)
SLOW_TENSION_PAIRS = [
    "saturn_uranus_tension", "jupiter_pluto_tension", "jupiter_uranus_tension",
    "saturn_neptune_tension", "jupiter_neptune_tension", "jupiter_saturn_tension",
]

def _rolling_percentile_rank(series: pd.Series, window: int = 1260, min_periods: int = 30) -> pd.Series:
    """
    Calcula el rango percentil móvil de forma vectorizada (sin lambdas lentos).
    Para cada punto, calcula qué fracción de los últimos `window` valores son <= al valor actual.
    """
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    
    for i in range(min_periods - 1, n):
        start = max(0, i - window + 1)
        window_data = values[start:i + 1]
        valid = window_data[~np.isnan(window_data)]
        if len(valid) >= min_periods:
            result[i] = stats.percentileofscore(valid, values[i], kind='rank')
    
    out = pd.Series(result, index=series.index, name=series.name)
    out = out.fillna(50.0) # [PARCHE]: Mediana neutra para evitar data leakage en primeros 30 días
    return out


def compute_slow_tension(tensor_df: pd.DataFrame, window_days: int = 1260) -> pd.Series:
    """
    Calcula la tensión estructural usando SOLO información del pasado.
    [OPTIMIZADO]: Usa _rolling_percentile_rank vectorizado.
    """
    slow_cols = [c for c in tensor_df.columns if any(s in c for s in SLOW_TENSION_PAIRS)]
    if not slow_cols:
        return pd.Series(0, index=tensor_df.index, name="slow_tension")
    raw = tensor_df[slow_cols].sum(axis=1)
    s = _rolling_percentile_rank(raw, window=window_days)
    s.name = "slow_tension"
    return s


def get_rolling_percentiles(series: pd.Series, window_days: int = 1260):
    """Calcula percentiles p50, p75, p90 usando solo datos pasados (rolling)."""
    p50 = series.rolling(window=window_days, min_periods=30).quantile(0.50)
    p75 = series.rolling(window=window_days, min_periods=30).quantile(0.75)
    p90 = series.rolling(window=window_days, min_periods=30).quantile(0.90)
    return p50.bfill(), p75.bfill(), p90.bfill()


def cascade_multiplier(tensor_df: pd.DataFrame, signals_df: pd.DataFrame,
                       out_dir: Path) -> pd.DataFrame:
    """
    Detecta ventanas de riesgo asimétrico: intersección de
    tensión estructural alta (>P75) + VIX elevado (>P75).
    
    En ecosistema sano: un spike AR5 se absorbe.
    En ecosistema frágil: un spike AR5 perfora el piso → cascada.
    """
    print("\n=== MULTIPLICADOR DE CASCADA ===")
    
    slow = compute_slow_tension(tensor_df)
    df = pd.DataFrame({"slow_tension": slow})
    df = df.join(signals_df[["vix"]]).dropna()
    
    # Percentiles dinámicos (rolling, sin look-ahead bias)
    _, slow_p75, _ = get_rolling_percentiles(df["slow_tension"])
    _, vix_p75, _  = get_rolling_percentiles(df["vix"])
    
    # Regímenes (comparando contra percentiles del pasado en cada día)
    df["regime_structural"] = (df["slow_tension"] > slow_p75).astype(int)
    df["regime_panic"]      = (df["vix"] > vix_p75).astype(int)
    df["cascade_risk"]      = df["regime_structural"] * df["regime_panic"]
    
    # Verificar si los días de cascada preceden a VIX extremo (>40)
    df["vix_future_max_5d"] = df["vix"].shift(-5).rolling(5, min_periods=1).max()
    
    # Ventanas de cascada
    cascade_days = df[df["cascade_risk"] == 1]
    n_total = len(df)
    n_cascade = len(cascade_days)
    pct = n_cascade / n_total * 100 if n_total > 0 else 0
    
    print(f"   Días totales analizados: {n_total}")
    print(f"   Días en CASCADA (tensión lenta alta + VIX alto): {n_cascade} ({pct:.1f}%)")
    
    if "vix" in df.columns:
        normal_days = df[df["cascade_risk"] == 0]
        cascade_with_spike = cascade_days[cascade_days["vix_future_max_5d"] > 40]
        normal_with_spike = normal_days[normal_days["vix_future_max_5d"] > 40] if len(normal_days) > 0 else pd.DataFrame()
        
        p_spike_cascade = len(cascade_with_spike) / max(1, n_cascade) * 100
        p_spike_normal  = len(normal_with_spike) / max(1, len(normal_days)) * 100
        
        print(f"\n   P(VIX>40 en 5 días | CASCADA): {p_spike_cascade:.2f}%")
        print(f"   P(VIX>40 en 5 días | NORMAL):  {p_spike_normal:.2f}%")
        if p_spike_normal > 0:
            print(f"   >>> Multiplicador de riesgo: {p_spike_cascade / p_spike_normal:.1f}x")
        else:
            print(f"   >>> Multiplicador de riesgo: ∞ (sin spikes en régimen normal)")
    
    df.to_csv(out_dir / "cascade_multiplier.csv")
    return df


def compute_supply_disruption(signals_df: pd.DataFrame) -> pd.Series:
    """
    Indicador Físico de Disrupción de Suministros basado en Commodities (Gold, Copper).
    Rastrea fluctuaciones absolutas como proxies de bottlenecks geopolíticos y escasez.
    """
    valid_cols = [c for c in ["gold", "copper"] if c in signals_df.columns]
    if not valid_cols:
        return pd.Series(0, index=signals_df.index)
        
    rets = signals_df[valid_cols].pct_change(20).abs()
    avg_anomaly = rets.mean(axis=1)
    
    disruption = _rolling_percentile_rank(avg_anomaly, window=1260)
    
    return disruption


def compute_network_alert(tensor_df: pd.DataFrame) -> pd.Series:
    """
    Crea el indicador 'Alerta de Disrupción de Redes' basado en los SHAP Values:
    Monitorea la combinación de Mercurio (Redes) con Urano/Júpiter/Neptuno.
    """
    # Pares clave identificados por SHAP
    network_cols = [
        "mercury_uranus_tension", 
        "mercury_jupiter_tension", 
        "mercury_neptune_tension"
    ]
    
    # Asegurarse de que las columnas existen en el tensor
    valid_cols = [c for c in network_cols if c in tensor_df.columns]
    if not valid_cols:
        return pd.Series(0, index=tensor_df.index)
        
    # Cálculo del score (promedio de tensiones de red)
    network_score = tensor_df[valid_cols].mean(axis=1)
    
    # Normalización por percentil móvil vectorizado
    alert_signal = _rolling_percentile_rank(network_score, window=1260)
    
    return alert_signal


# ─────────────────────────────────────────────
# 5d. MOTOR DEFCON (ESTADOS DE AMENAZA)
# ─────────────────────────────────────────────

DEFCON_LEVELS = {
    5: {"name": "DEFCON 5 — PEACE",    "color": "🟢", "desc": "Expansión máxima. Tensión baja, VIX bajo."},
    4: {"name": "DEFCON 4 — ELEVATED", "color": "🔵", "desc": "Tensión estructural subiendo. Monitoreo pasivo."},
    3: {"name": "DEFCON 3 — ALERT",    "color": "🟡", "desc": "Tensión alta O VIX elevado. Reducir exposición."},
    2: {"name": "DEFCON 2 — CRITICAL", "color": "🟠", "desc": "Tensión alta Y VIX elevado. Activar protocolos de preservación."},
    1: {"name": "DEFCON 1 — CASCADE",  "color": "🔴", "desc": "Intersección máxima. Zero-Day inminente. Ejecutar despliegue."},
}

def compute_defcon(tensor_df: pd.DataFrame, signals_df: pd.DataFrame,
                   out_dir: Path) -> pd.DataFrame:
    """
    Clasifica cada día en un nivel DEFCON (1-5) basado en:
    - Tensión estructural lenta (pares generacionales)
    - Régimen del VIX (nivel de pánico del mercado)
    - Riesgo de cascada (intersección de ambos)
    """
    print("\n" + "=" * 60)
    print("       MOTOR DEFCON — ESTADOS DE AMENAZA")
    print("=" * 60)
    
    slow = compute_slow_tension(tensor_df)
    df = pd.DataFrame({"slow_tension": slow})
    df = df.join(signals_df[["vix"]]).dropna()
    
    # Percentiles rolling (sin look-ahead bias)
    slow_p50, slow_p75, slow_p90 = get_rolling_percentiles(df["slow_tension"])
    vix_p50, vix_p75, _          = get_rolling_percentiles(df["vix"])
    
    def classify_defcon(idx):
        s = df.loc[idx, "slow_tension"]
        v = df.loc[idx, "vix"]
        sp90, sp75, sp50 = slow_p90.loc[idx], slow_p75.loc[idx], slow_p50.loc[idx]
        vp75, vp50 = vix_p75.loc[idx], vix_p50.loc[idx]
        if s > sp90 and v > vp75:
            return 1  # CASCADE
        elif s > sp75 and v > vp75:
            return 2  # CRITICAL
        elif s > sp75 or v > vp75:
            return 3  # ALERT
        elif s > sp50 or v > vp50:
            return 4  # ELEVATED
        else:
            return 5  # PEACE
    
    df["defcon"] = [classify_defcon(idx) for idx in df.index]
    
    # Distribución histórica
    print("\n   Distribución histórica de niveles DEFCON:")
    for level in range(1, 6):
        info = DEFCON_LEVELS[level]
        count = (df["defcon"] == level).sum()
        pct = count / len(df) * 100
        print(f"   {info['color']} {info['name']}: {count:>6} días ({pct:.1f}%) — {info['desc']}")
    
    network_alert_series = compute_network_alert(tensor_df)
    
    # Estado actual (últimos 7 días) con el nuevo indicador
    print("\n   ─── ESTADO ACTUAL (últimos 7 días) ───")
    last_week = df.tail(7)
    for date, row in last_week.iterrows():
        level = int(row["defcon"])
        info = DEFCON_LEVELS[level]
        net_val = network_alert_series.loc[date]
        
        # Alerta visual si la red está en zona crítica (>P90)
        net_tag = "⚠️ CRÍTICO" if net_val > 90 else "NORMAL"
        
        print(f"   {date.strftime('%Y-%m-%d')} | {info['color']} {info['name']} | "
              f"Tensión: {row['slow_tension']:.1f} | VIX: {row['vix']:.1f} | Redes: {net_val:.1f}% ({net_tag})")
    
    # Estado dominante actual
    current_defcon = int(last_week["defcon"].mode().iloc[0])
    current_info = DEFCON_LEVELS[current_defcon]
    print(f"\n   {'=' * 50}")
    print(f"   {current_info['color']}  ESTADO DOMINANTE: {current_info['name']}")
    print(f"   {current_info['desc']}")
    print(f"   {'=' * 50}")
    
    df.to_csv(out_dir / "defcon_states.csv")
    return df


# ─────────────────────────────────────────────
# 5e. DEEP SCAN A: SEGMENTACIÓN POR RÉGIMEN (BULL VS BEAR)
# ─────────────────────────────────────────────

def deep_scan_regime_segmentation(ml_dataset: dict, signals_df: pd.DataFrame, out_dir: Path):
    """
    Analiza la influencia planetaria segmentando entre regímenes de VIX.
    VIX < 20 (Calma/Bull) vs VIX > 30 (Pánico/Bear).
    """
    print("\n=== DEEP SCAN A: SEGMENTACIÓN POR RÉGIMEN ===")
    
    X_train = ml_dataset["X_plan_train"]
    baseline_cols = [c for c in X_train.columns if "vix" in c] # Incluye vix y vix_lag_X
    
    # Unir el target Delta de signals_df para no superponer columnas y obligar a predecir aceleración
    target_series = (signals_df["vix"].shift(-1) - signals_df["vix"]).rename("target_vix")
    df = X_train.join(target_series).dropna()
    
    bull_df = df[df["vix"] < 20]
    bear_df = df[df["vix"] > 30]
    
    print(f"   Días Bull (VIX < 20): {len(bull_df)}")
    print(f"   Días Bear (VIX > 30): {len(bear_df)}")
    
    def evaluate_regime(regime_df, name):
        if len(regime_df) < 50:
            return
        X = regime_df.drop(columns=["target_vix"]) # Mantiene "vix" en features para ancla
        y = regime_df["target_vix"]
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        ridge = RidgeCV(alphas=np.logspace(-2, 4, 30))
        ridge.fit(X_scaled, y)
        
        coefs = pd.DataFrame({
            "feature": X.columns,
            "weight": ridge.coef_
        })
        coefs = coefs[~coefs["feature"].isin(baseline_cols)].copy()
        coefs["abs_weight"] = coefs["weight"].abs()
        top = coefs.sort_values("abs_weight", ascending=False).head(5)
        print(f"\n   Top Features en Régimen {name}:")
        print(top[["feature", "weight"]].to_string(index=False))
        top.to_csv(out_dir / f"deep_scan_A_top_features_{name}.csv", index=False)
        
    evaluate_regime(bull_df, "BULL")
    evaluate_regime(bear_df, "BEAR")


# ─────────────────────────────────────────────
# 5f. DEEP SCAN B: EVENTOS FÍSICOS GDELT
# ─────────────────────────────────────────────

def deep_scan_gdelt_events(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, out_dir: Path):
    """
    Correlaciona las tensiones planetarias directamente con picos de eventos físicos (GDELT).
    """
    print("\n=== DEEP SCAN B: EVENTOS FÍSICOS GDELT ===")
    
    if "gdelt_conflict" not in signals_df.columns:
        print("   No hay datos GDELT disponibles.")
        return
        
    # Identificar picos GDELT (> P80)
    gdelt = signals_df["gdelt_conflict"].dropna()
    p80 = gdelt.quantile(0.80)
    gdelt_spikes = gdelt[gdelt > p80]
    
    print(f"   Días con alta intensidad de eventos físicos GDELT (> P80): {len(gdelt_spikes)}")
    
    if len(gdelt_spikes) < 10:
        return
        
    # Tomar la tensión generacional en esos días
    slow = compute_slow_tension(tensor_df)
    df = pd.DataFrame({"slow_tension": slow, "gdelt": signals_df["gdelt_conflict"]}).dropna()
    
    # Correlación simple en extremos
    corr = df.corr().iloc[0, 1]
    print(f"   Correlación lineal Tensión-GDELT general: {corr:.3f}")
    
    # Análisis de tensión promedio en días de picos GDELT vs días normales
    spike_idx = gdelt_spikes.index.intersection(df.index)
    avg_tension_spike = df.loc[spike_idx, "slow_tension"].mean() if len(spike_idx) > 0 else np.nan
    normal_days = df[df["gdelt"] <= p80]
    avg_tension_normal = normal_days["slow_tension"].mean() if len(normal_days) > 0 else np.nan
    
    print(f"   Tensión Lenta Promedio en Picos GDELT:   {avg_tension_spike:.1f}")
    print(f"   Tensión Lenta Promedio en Días Normales: {avg_tension_normal:.1f}")
    
    pd.DataFrame({"metric": ["corr", "avg_tension_spike", "avg_tension_normal"],
                  "value": [corr, avg_tension_spike, avg_tension_normal]}).to_csv(out_dir / "deep_scan_B_gdelt.csv", index=False)


# ─────────────────────────────────────────────
# 5g. DEEP SCAN C: FOREST SCAN (NO-LINEALIDAD CON XGBOOST Y SHAP)
# ─────────────────────────────────────────────

def deep_scan_forest_shap(ml_dataset: dict, out_dir: Path):
    """
    Entrena un XGBoost y usa SHAP para encontrar interacciones planetarias no lineales.
    """
    print("\n=== DEEP SCAN C: FOREST SCAN (XGBoost + SHAP) ===")
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        print("   [!] xgboost o shap no están instalados. Omítiendo Deep Scan C.")
        return
        
    X_train, y_train = ml_dataset["X_plan_train"], ml_dataset["y_clf_train"]
    
    if len(X_train) == 0:
        return
        
    print("   Entrenando XGBoost Classifier para VIX > 30 (buscando no-linealidades)...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, 
        random_state=42, eval_metric='logloss', n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("   Calculando SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    
    # Para ahorrar memoria, usar una muestra aleatoria de hasta 2000 días
    if len(X_train) > 2000:
        X_sample = shap.sample(X_train, 2000, random_state=42)
    else:
        X_sample = X_train
        
    shap_values = explainer.shap_values(X_sample)
    
    # Calcular la importancia media absoluta de SHAP
    if isinstance(shap_values, list): # Dependiendo la versión de SHAP, clasificación binaria puede devolver list de 2 arrays
        shap_values = shap_values[1]
        
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)), columns=['feature','shap_importance'])
    feature_importance.sort_values(by=['shap_importance'], ascending=False, inplace=True)
    
    baseline_cols = [c for c in X_train.columns if "vix_lag" in c]
    planets_only = feature_importance[~feature_importance['feature'].isin(baseline_cols)].copy()
    
    top = planets_only.head(10)
    print("\n   Top 10 Features Planetarios (Importancia No-Lineal SHAP):")
    print(top.to_string(index=False))
    top.to_csv(out_dir / "deep_scan_C_shap_top_features.csv", index=False)


# ─────────────────────────────────────────────
# 5h. DEEP SCAN D: CORRELACIÓN CON LA MATERIA
# ─────────────────────────────────────────────

def deep_scan_commodities_correlation(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, out_dir: Path):
    """
    Analiza la correlación resurgida del borde Físico de Materiales Orgánicos:
    - Oro (gold) <-> Sol/Venus
    - Cobre (copper) <-> Saturno/Urano
    """
    print("\n=== DEEP SCAN D: CORRELACIÓN CON LA MATERIA (ORO Y COBRE) ===")
    
    valid_comms = [c for c in ["gold", "copper"] if c in signals_df.columns]
    if len(valid_comms) < 2:
        print("   [!] Faltan datos de commodities. Deep Scan D abortado.")
        return
        
    pairs_map = {
        "gold": "sun_venus_tension",
        "copper": "saturn_uranus_tension"
    }
    
    df = signals_df[valid_comms].copy()
    
    for comm, pair in pairs_map.items():
        if pair in tensor_df.columns:
            df[pair] = tensor_df[pair]
        else:
            df[pair] = 0.0
            
    for comm in valid_comms:
        df[f"{comm}_ret_30d"] = df[comm].rolling(30, min_periods=1).max().shift(-30) / df[comm] - 1
        
    df.dropna(inplace=True)
    
    for comm, pair in pairs_map.items():
        corr = df[pair].corr(df[f"{comm}_ret_30d"])
        print(f"\n   --- Análisis de {comm.upper()} vs {pair.upper()} ---")
        print(f"   Correlación lineal Tensión-MFE(30d): {corr:.3f}")
        
        # [PARCHE]: Evitar Look-Ahead Bias calculando P90 móvil de últimos 5 años (1260 días)
        pair_p90 = _rolling_percentile_rank(df[pair], window=1260)
        tension_alta = df[pair_p90 > 90]
        tension_normal = df[(pair_p90 <= 90) & pair_p90.notna()]
        
        spike_alta = tension_alta[tension_alta[f"{comm}_ret_30d"] > 0.05]
        spike_normal = tension_normal[tension_normal[f"{comm}_ret_30d"] > 0.05]
        
        p_spike_alta = len(spike_alta) / max(1, len(tension_alta)) * 100
        p_spike_normal = len(spike_normal) / max(1, len(tension_normal)) * 100
        
        mult = p_spike_alta / max(1e-9, p_spike_normal) if p_spike_normal > 0 else float('inf')
        print(f"   P({comm} sube > 5% en 30d | NORMAL)           : {p_spike_normal:.2f}%")
        print(f"   P({comm} sube > 5% en 30d | TENSIÓN > P90)    : {p_spike_alta:.2f}% (>>> Multiplicador de Escasez: {mult:.2f}x)")


# ─────────────────────────────────────────────
# 5i. MOTOR GANN: GEOMETRÍA DE LA MATERIA (ORO Y COBRE)
# ─────────────────────────────────────────────

def compute_gann_engine(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, out_dir: Path):
    """
    W.D. Gann: Cruza la tensión planetaria lenta con la acción del precio.
    Busca intersecciones donde el activo está en un mínimo técnico pero la tensión física está en rojo,
    indicando un "Piso Geométrico" (Soporte).
    [MEJORADO]: Percentil P90 rolling para evitar look-ahead bias.
    """
    print("\n=== MOTOR GANN: GEOMETRÍA DE PRECIO Y TIEMPO ===")
    
    valid_comms = [c for c in ["gold", "copper"] if c in signals_df.columns]
    if len(valid_comms) < 2:
        print("   [!] Faltan datos de commodities. Motor Gann abortado.")
        return
        
    df = signals_df[valid_comms].copy()
    df["sun_venus"] = tensor_df.get("sun_venus_tension", 0)
    df["saturn_uranus"] = tensor_df.get("saturn_uranus_tension", 0)
    
    # Geometría del Precio: Bollinger Band Inferior 50 días (sobrevendido extremo)
    for comm in valid_comms:
        df[f"{comm}_ma50"] = df[comm].rolling(50).mean()
        df[f"{comm}_std50"] = df[comm].rolling(50).std()
        df[f"{comm}_lower_band"] = df[f"{comm}_ma50"] - (2 * df[f"{comm}_std50"])
        df[f"{comm}_is_low"] = (df[comm] < df[f"{comm}_lower_band"]).astype(int)
    
    # Geometría del Tiempo: Rolling P90 (sin look-ahead)
    _, _, gold_p90 = get_rolling_percentiles(df["sun_venus"])
    _, _, copper_p90 = get_rolling_percentiles(df["saturn_uranus"])
    
    # Señales de Compra Gann (Piso Estructural Extremo)
    df["gann_gold_buy"] = ((df["sun_venus"] > gold_p90) & (df["gold_is_low"] == 1)).astype(int)
    df["gann_copper_buy"] = ((df["saturn_uranus"] > copper_p90) & (df["copper_is_low"] == 1)).astype(int)
    
    gold_signals = df["gann_gold_buy"].sum()
    copper_signals = df["gann_copper_buy"].sum()
    
    print(f"   Señales históricas de Piso en Oro:   {gold_signals} días")
    print(f"   Señales históricas de Piso en Cobre: {copper_signals} días")
    
    # Verificar rendimiento post-señal (Maximum Favorable Excursion a 30 días)
    for comm, col in [("gold", "gann_gold_buy"), ("copper", "gann_copper_buy")]:
        df[f"{comm}_ret_30d"] = df[comm].rolling(30, min_periods=1).max().shift(-30) / df[comm] - 1
        signal_days = df[df[col] == 1]
        normal_days = df[(df[f"{comm}_is_low"] == 1) & (df[col] == 0)] # Aislar el verdadero Alpha
        if len(signal_days) > 0:
            avg_ret_signal = signal_days[f"{comm}_ret_30d"].mean() * 100
            avg_ret_normal = normal_days[f"{comm}_ret_30d"].mean() * 100
            print(f"   >>> MFE promedio 30d después de señal Gann {comm.upper()}: {avg_ret_signal:.2f}% vs Normal: {avg_ret_normal:.2f}%")
    
    # Estado actual
    last_day = df.iloc[-1]
    print("\n   --- SEÑALES GANN ACTUALES ---")
    if last_day["gann_gold_buy"] == 1:
        print(f"   [ALERTA ORO] 🟡 Configuración Gann Activa: Precio bajo Bollinger 50d + Tensión Sol-Venus P90.")
    else:
        print(f"   [ORO] Normal. Precio: {last_day['gold']:.2f} | B-Inferior(50d): {last_day['gold_lower_band']:.2f}")
        
    if last_day["gann_copper_buy"] == 1:
        print(f"   [ALERTA COBRE] 🟠 Configuración Gann Activa: Precio bajo Bollinger 50d + Tensión Saturno-Urano P90.")
    else:
        print(f"   [COBRE] Normal. Precio: {last_day['copper']:.4f} | B-Inferior(50d): {last_day['copper_lower_band']:.4f}")

    df.to_csv(out_dir / "gann_engine_signals.csv")
    return df


# ─────────────────────────────────────────────
# 5j. OSCILADOR LUNAR: HIGH-FREQUENCY VIX SWING TRADING
# ─────────────────────────────────────────────

def compute_lunar_oscillator(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, out_dir: Path):
    """
    Efecto Lunar: Swing Trading a corto plazo.
    Picos biológicos (Luna Nueva/Llena = Tensión Sol-Luna Máxima)
    coincidiendo con euforia o pánico en el VIX para predecir reversiones a la media.
    [MEJORADO]: Bollinger rolling, backtesting de la señal.
    """
    print("\n=== OSCILADOR LUNAR: REVERSIÓN BIOLÓGICA (VIX) ===")
    
    if "sun_moon_tension" not in tensor_df.columns or "vix" not in signals_df.columns:
        print("   [!] Faltan datos requeridos. Oscilador Lunar abortado.")
        return
        
    df = signals_df[["vix"]].copy()
    df["sun_moon"] = tensor_df["sun_moon_tension"]
    
    # Bollinger Bands rolling para el VIX
    df["vix_ma10"] = df["vix"].rolling(10).mean()
    df["vix_std10"] = df["vix"].rolling(10).std()
    df["vix_upper"] = df["vix_ma10"] + (1.5 * df["vix_std10"])
    
    # Picos lunares con Rolling P90 (sin look-ahead)
    _, _, lunar_p90 = get_rolling_percentiles(df["sun_moon"])
    
    # Señal: VIX sobre Bollinger superior Y pico biológico
    df["lunar_short_vix"] = ((df["vix"] > df["vix_upper"]) & (df["sun_moon"] > lunar_p90)).astype(int)
    
    signals_count = df["lunar_short_vix"].sum()
    print(f"   Señales históricas de Corto en VIX (Agotamiento Lunar): {signals_count}")
    
    # Backtest: ¿Qué pasa 5 días después de la señal?
    df["vix_ret_5d"] = df["vix"].shift(-5) / df["vix"] - 1
    signal_days = df[df["lunar_short_vix"] == 1]
    
    # Baseline riguroso: VIX sobre banda SUP *pero* la biología está en calma
    normal_days = df[(df["vix"] > df["vix_upper"]) & (df["sun_moon"] <= lunar_p90)]
    
    if len(signal_days) > 0:
        avg_ret_signal = signal_days["vix_ret_5d"].mean() * 100
        avg_ret_normal = normal_days["vix_ret_5d"].mean() * 100
        print(f"   >>> Cambio promedio VIX 5d después de señal Lunar: {avg_ret_signal:.2f}% vs Normal (Puro Técnico): {avg_ret_normal:.2f}%")
    
    # Estado actual
    last_day = df.iloc[-1]
    print("\n   --- ESTADO OSCILADOR LUNAR ACTUAL ---")
    if last_day["lunar_short_vix"] == 1:
        print("   [ALERTA SWING] 📉 Oportunidad SHORT VIX. Pánico sobre-extendido coincidente con pico lunar.")
    else:
        print(f"   [SWING] Sin señal. VIX: {last_day['vix']:.2f} | Banda Sup: {last_day['vix_upper']:.2f}")

    df.to_csv(out_dir / "lunar_oscillator.csv")
    return df


# ─────────────────────────────────────────────
# 5k. MOTOR MACRO: ASIGNACIÓN DE PORTAFOLIO DEFCON
# ─────────────────────────────────────────────

def compute_macro_allocation(defcon_df: pd.DataFrame, out_dir: Path):
    """
    Automatiza la distribución de capital basado en el riesgo estructural DEFCON.
    Distribuye 100% del capital entre: Cash, Oro, Cobre, Equities.
    """
    print("\n=== MOTOR MACRO: ASIGNACIÓN DE PORTAFOLIO ===")
    
    allocation_rules = {
        1: {"Cash": 80, "Gold": 20, "Copper": 0,  "Equities": 0},   # CASCADE
        2: {"Cash": 40, "Gold": 40, "Copper": 0,  "Equities": 20},  # CRITICAL
        3: {"Cash": 20, "Gold": 30, "Copper": 20, "Equities": 30},  # ALERT
        4: {"Cash": 10, "Gold": 15, "Copper": 35, "Equities": 40},  # ELEVATED
        5: {"Cash": 5,  "Gold": 5,  "Copper": 20, "Equities": 70},  # PEACE
    }
    
    df = defcon_df.copy()
    for asset in ["Cash", "Gold", "Copper", "Equities"]:
        df[f"alloc_{asset}"] = df["defcon"].map(lambda x, a=asset: allocation_rules[int(x)][a])
    
    last_date = df.index[-1].strftime("%Y-%m-%d")
    current_defcon = int(df.iloc[-1]["defcon"])
    alloc = allocation_rules[current_defcon]
    
    print(f"\n   Recomendación Oficial para {last_date} (DEFCON {current_defcon}):")
    print(f"   💵 Cash:       {alloc['Cash']}%")
    print(f"   🟡 Oro:        {alloc['Gold']}%")
    print(f"   🟠 Cobre:      {alloc['Copper']}%")
    print(f"   📈 Acciones:   {alloc['Equities']}%")
    
    df.to_csv(out_dir / "macro_portfolio_allocation.csv")
    return df


# ─────────────────────────────────────────────
# 6. ANÁLISIS DE FECHA ÚNICA
# ─────────────────────────────────────────────

def analyze_single_date(date_str: str):
    """Análisis completo de tensión para una fecha específica."""
    print(f"\n=== ANÁLISIS DE FECHA: {date_str} ===\n")
    eph, ts = load_ephemeris()

    d = datetime.strptime(date_str, "%Y-%m-%d")
    t = ts.utc(d.year, d.month, d.day)

    from skyfield.api import Loader
    planet_bodies = {
        "sun":     eph["sun"],
        "moon":    eph["moon"],
        "mercury": eph["mercury barycenter"],
        "venus":   eph["venus barycenter"],
        "mars":    eph["mars barycenter"],
        "jupiter": eph["jupiter barycenter"],
        "saturn":  eph["saturn barycenter"],
        "uranus":  eph["uranus barycenter"],
        "neptune": eph["neptune barycenter"],
        "pluto":   eph["pluto barycenter"],
    }

    positions = {}
    for key, body in planet_bodies.items():
        positions[key] = get_ecliptic_lon(body, t, eph)

    print(f"{'Planeta':<12} {'Longitud':>10}°  Signo")
    print("-" * 40)
    signs = ["Aries","Tauro","Géminis","Cáncer","Leo","Virgo",
             "Libra","Escorpio","Sagitario","Capricornio","Acuario","Piscis"]
    for key, lon in positions.items():
        sign = signs[int(lon // 30)]
        print(f"{PLANETS[key]['name']:<12} {lon:>10.2f}°  {sign}")

    print("\n=== ASPECTOS ACTIVOS (por tensión) ===\n")
    aspects_found = []
    planet_keys = list(positions.keys())

    for i in range(len(planet_keys)):
        for j in range(i+1, len(planet_keys)):
            p1, p2 = planet_keys[i], planet_keys[j]
            w1 = PLANETS[p1]["weight"]
            w2 = PLANETS[p2]["weight"]
            res = compute_aspect_tension(positions[p1], positions[p2], w1, w2)
            for asp in res["active"]:
                aspects_found.append({
                    "par": f"{PLANETS[p1]['name']} — {PLANETS[p2]['name']}",
                    "aspecto": asp["aspect"],
                    "tensión": asp["tension"],
                    "orbe": asp["orb_real"],
                })

    aspects_found.sort(key=lambda x: -x["tensión"])

    if aspects_found:
        print(f"{'Par':<35} {'Aspecto':<14} {'Tensión':>8}  {'Orbe':>6}")
        print("-" * 70)
        for a in aspects_found[:15]:
            print(f"{a['par']:<35} {a['aspecto']:<14} {a['tensión']:>8.3f}  {a['orbe']:>5.2f}°")
    else:
        print("No hay aspectos activos con los orbes configurados.")

    # Nodo puente
    node_scores = {}
    for a in aspects_found:
        for name in a["par"].split(" — "):
            node_scores[name] = node_scores.get(name, 0) + a["tensión"]

    if node_scores:
        bridge = max(node_scores, key=node_scores.get)
        print(f"\n>>> Nodo puente: {bridge} (tensión acumulada: {node_scores[bridge]:.3f})")

    total_raw  = sum(a["tensión"] for a in aspects_found)
    max_poss   = 45 * 1.6 * 2.0
    global_pct = min(100, total_raw / max_poss * 100)
    print(f">>> Tensión global: {global_pct:.1f}%")

    return aspects_found


# ─────────────────────────────────────────────
# 7. PIPELINE COMPLETO
# ─────────────────────────────────────────────

def run_full_pipeline(start_date: str, end_date: str):
    """Ejecuta el pipeline completo y guarda los resultados."""
    # Crear carpeta de salida con fechas de rango y timestamp de ejecución
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_label = f"{start_date}_to_{end_date}_{run_timestamp}"
    out_dir = Path("./output") / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Carpeta de salida: {out_dir.resolve()}/\n")

    # 1. Posiciones planetarias
    positions_df = compute_daily_positions(start_date, end_date)
    positions_df.to_csv(out_dir / f"positions_{run_label}.csv")
    print(f"   Guardado: positions_{run_label}.csv ({len(positions_df)} días)")

    # 2. Tensor de tensión
    tensor_df = build_tensor(positions_df)
    tensor_df.to_csv(out_dir / f"tensor_{run_label}.csv")
    print(f"   Guardado: tensor_{run_label}.csv ({len(tensor_df.columns)} columnas)")

    # 3. Señales externas e indicadores físicos
    vix   = fetch_vix(start_date, end_date)
    gdelt = fetch_gdelt_conflict(start_date, end_date)
    comms = fetch_commodities(start_date, end_date)

    signals_df = pd.DataFrame({"vix": vix, "gdelt_conflict": gdelt}).join(comms)
    signals_df.to_csv(out_dir / f"signals_{run_label}.csv")
    print(f"   Guardado: signals_{run_label}.csv")

    # 4. Pipeline Machine Learning Estratégico (Lag, Train/Test, Baseline, Ridge)
    ml_dataset = build_ml_dataset(tensor_df, signals_df, max_lag=30)
    metrics, test_dates = train_evaluate_models(ml_dataset, out_dir)

    # 5. Motor Estratégico: Cascada + DEFCON
    cascade_multiplier(tensor_df, signals_df, out_dir)
    defcon_df = compute_defcon(tensor_df, signals_df, out_dir)
    
    # 6. Deep Scans (No-Linealidad, Regímenes, Eventos Físicos)
    deep_scan_regime_segmentation(ml_dataset, signals_df, out_dir)
    deep_scan_gdelt_events(tensor_df, signals_df, out_dir)
    deep_scan_forest_shap(ml_dataset, out_dir)
    deep_scan_commodities_correlation(tensor_df, signals_df, out_dir)
    
    # 7. Motores Tácticos (Gann + Lunar + Macro)
    compute_gann_engine(tensor_df, signals_df, out_dir)
    compute_lunar_oscillator(tensor_df, signals_df, out_dir)
    compute_macro_allocation(defcon_df, out_dir)

    # 7. Reporte de convergencias generales (para exploración visual)
    merged = tensor_df[["global_tension"]].join(signals_df)
    convergences = merged[
        (merged["global_tension"] > 70) & (merged["vix"] > VIX_CRITICAL)
    ]
    if len(convergences) > 0:
        convergences.to_csv(out_dir / f"convergences_{run_label}.csv")

    print(f"\nResultados guardados en {out_dir.resolve()}/")
    return tensor_df, signals_df, out_dir


# ─────────────────────────────────────────────
# 8. VISUALIZACIÓN RÁPIDA (ACTUALIZADA)
# ─────────────────────────────────────────────

def plot_results(tensor_df: pd.DataFrame, signals_df: pd.DataFrame, out_dir: Path = None):
    """Genera gráficos: tensión clásica y rendimiento del modelo Ridge en Out-Of-Sample."""
    if out_dir is None:
        out_dir = Path("./output")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig = plt.figure(figsize=(16, 6))
    
    # — Gráfico 1: Tensión global vs VIX —
    ax1 = fig.add_subplot(111)
    merged = tensor_df[["global_tension"]].join(signals_df[["vix"]]).dropna()
    ax2 = ax1.twinx()

    ax1.plot(merged.index, merged["global_tension"],
             color="#7F77DD", lw=1.5, label="Tensión planetaria (%)")
    ax2.plot(merged.index, merged["vix"],
             color="#E24B4A", lw=1.2, alpha=0.8, label="VIX")
    ax1.axhline(70, color="#7F77DD", ls="--", lw=0.8, alpha=0.5)
    ax2.axhline(VIX_CRITICAL, color="#E24B4A", ls="--", lw=0.8, alpha=0.5, label=f"VIX umbral {VIX_CRITICAL}")

    ax1.set_ylabel("Tensión planetaria (%)", color="#7F77DD")
    ax2.set_ylabel("VIX", color="#E24B4A")
    ax1.set_title("Tensión planetaria compuesta vs VIX", fontsize=13, fontweight="bold")
    lines1, labs1 = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labs1 + labs2, loc="upper left", fontsize=9)

    chart_path = out_dir / f"planetary_analysis_{out_dir.name}.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    print(f"   Gráfico guardado: {chart_path}")
    plt.show()


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Motor de análisis planetario universal")
    parser.add_argument("--start", default="2010-01-01", help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2024-01-01", help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--date",  default=None, help="Análisis de fecha única (YYYY-MM-DD)")
    parser.add_argument("--no-plot", action="store_true", help="Omitir visualización")
    args = parser.parse_args()

    if args.date:
        analyze_single_date(args.date)
    else:
        tensor_df, signals_df, out_dir = run_full_pipeline(args.start, args.end)
        if not args.no_plot:
            try:
                plot_results(tensor_df, signals_df, out_dir)
            except ImportError:
                print("   (instala matplotlib y seaborn para ver gráficos)")


if __name__ == "__main__":
    main()
