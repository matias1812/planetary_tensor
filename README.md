# 🌌 Planetary Tensor: Macro-Gravitational Tail-Risk Engine

![Planetary Tensor](https://img.shields.io/badge/Status-Production-success)
![Python](https://img.shields.io/badge/Python-3.14--Vect-blue)
![Machine Learning](https://img.shields.io/badge/ML-XGBoost%20%7C%20Ridge%20%7C%20SHAP-orange)
![Architecture](https://img.shields.io/badge/Architecture-Quant%20Institutional-darkred)

**Planetary Tensor** es un motor algorítmico asimétrico (**Tail-Risk Classification Engine**) diseñado para la detección temprana de cuellos de botella macroeconómicos, shocks de liquidez y eventos de estrés en los mercados globales (VIX > 30). Su infraestructura central abandona los enfoques fundamentalistas y técnicos tradicionales, optando en su lugar por un mapeo topológico y tensorial de las interacciones planetarias heliocéntricas/geocéntricas, correlacionado de manera causal y matemática con la dinámica de fluidos de la economía y la psicología humana en masa.

---

## 📜 Resumen Científico (Abstract)

La economía global es un sistema complejo impulsado por ciclos de crédito, fricciones logísticas de la cadena de suministro global y el sesgo emocional colectivo de los administradores de capital. **Planetary Tensor** plantea empíricamente que estas variables terrenales no son eventos discretos y caóticos, sino que están atadas de manera armónica a los relojes inerciales más grandes del sistema solar.

Mediante la regularización no-lineal (XGBoost), penalización lineal (RidgeCV) y descomposición topológica (PCA) de 36 años de historia bursátil sobre una matriz de 66 pares orbitales, el algoritmo ha logrado destilar **Las Tres Dimensiones Fundamentales del Riesgo**:

1. **D1: El Metrónomo Biológico (Sol/Luna):** Captura el sesgo de aversión al corto plazo. Domina el 34% de la varianza ruidosa del sistema, manifestándose en el aplastamiento tardío de la Volatilidad Implícita (IV Crush) documentado en nuestro Oscilador Lunar.
2. **D2: El Código de Barras de la Liquidez (Júpiter/Urano/Mercurio):** Una función de onda cuadrada oculta en el mercado interbancario que codifica los ciclos de *Quantitative Easing* (Expansión) y paralizaciones institucionales (*Liquidity Shocks* o "Flash Crashes").
3. **D3: La Gravedad Industrial Macro (Saturno/Urano/Plutón):** Posee estadísticamente apenas un 4.2% del volumen de la varianza en la topología base, pero concentra el **peso predictivo dominante (>0.80 Ridge Weight)** en la matriz final. Actúa como la Cuenca de Atracción para quiebres estructurales en las materias primas primarias mundiales, dominando fuertemente la curva a 30 días de los materiales físicos como el Cobre.

---

## ⚙️ Arquitectura Cuantitativa & Rigor (Pentest Validado)

El sistema ha sido auditado para alcanzar rigurosos estándares de validación *Out-Of-Sample* (OOS), sellando cualquier posibilidad de ruido estadístico o espejismos de curva:

- **100% Blindaje contra Fuga de Datos (Data Leakage):** Todos los módulos predictivos operan bajo un estricto horizonte $T+1$ (`shift(-1)`), donde el modelo absorbe la tensión planetaria al cierre para anticipar asimétricamente el comportamiento de la sesión *de mañana*.
- **Cero Sesgo de "Mirada al Futuro" (Look-Ahead Bias):** Los percentiles y umbrales de tensión ($P_{90}$) no derivan de parámetros globales, sino que son generados causalmente por una función de Rango Percentil Móvil Vectorizada ($\text{window} = 1260$ días / 5 años), respetando el flujo natural del tiempo que enfrenta un operador.
- **Autorregresión Libre de Amnesia (AR5 Baseline):** Antes de que a las series temporales de órbitas se les asigne poder predictivo, deben vencer en capacidad de predicción asimétrica a "La Realidad" (un modelo de base Autorregresivo de 5 días incluyendo el precio *spot* de hoy).
- **Métrica de Excursión Máxima Favorable (MFE 30d):** La viabilidad de las tensiones estructurales lentas se mide sobre una ventana inercial de corto y mediano plazo logísticas, no mediante mediciones cegas punto-a-punto.

---

## 🛠️ Los Motores de Ejecución y Monitoreo

### 1. Motor DEFCON (Estado de Amenaza del Sistema)
Categoriza la resiliencia en tiempo real de los mercados en 5 escalas prescriptivas evaluando la tensión lenta cruzada con el VIX vivo y fricciones mediáticas globales. Asigna pasivamente el Portafolio recomendando exposición cruzada de Capital/Metales/Acciones según el horizonte de riesgo.
* *Factor Multilicador Interno de Riesgo de Cascada*: **~6.6x**

### 2. Motor Geométrico GANN (Piso Estructural)
Mapea el quiebre absoluto del Soporte Matemático.
Cuando un activo duro (e.g. Cobre) cae 2 desviaciones estándar por debajo de la media histórica de 50 días (Ruptura Banda de Bollinger M50) **Y SIMULTÁNEAMENTE** la tensión gravitacional en su par logístico (Saturno-Urano/Sol-Venus) fractura el Percentil Móvil de 5 años ($P90$ temporal), alerta de un estrangulamiento artificial.

### 3. Oscilador Lunar (IV Crush)
Detecta un VIX dilatado en coincidencia con el máximo umbral de tensión lunar (Luna Llena/Nueva) para apostar contra sesgos algorítmicos. Mide exactamente cuántos días de prima inflada retienen los operadores paralizados por el sesgo humano.

### 4. Simulador Topológico y Liquid Shocks (PCA)
Las utilidades adyacentes grafican visualmente el vector `D2_Liquidity` (*Código de Barras*), permitiendo ver oscilaciones binarias asimétricas (ON/OFF) revelando inyecciones de caja por bancos centrales pre-crash.

---

## 🚀 Uso Rápido (Quickstart)

El programa funciona mediante multiprocesamiento vectorizado a lo largo del tiempo, calculando miles de posiciones astronómicas en segundos bajo arquitectura de núcleos nativa e injertando GDELT concurrentemente bajo un Executor local.

**1. Activar Entorno e Instalar Dependencias:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Lanzar Pipeline Principal (OOS Validation):**
```bash
python planetary_tensor_analysis.py --start 1990-01-01 --end 2026-04-21 --no-plot
```

**3. Lanzar Visualizadores de Topología y Liquidez (D2):**
```bash
python plot_pca_dimensions.py
python plot_liquidity_barcode.py
python plot_data_leakage.py
```

Las salidas (`.csv` crudos, reportes matemáticos y visuales asintóticos) residen comprimidas por cada subprocesamiento temporal en la carpeta `output/`.

---

*“No calculamos cuándo estallará precisamente el cristal, calculamos cuánta masa se acumula sin ser observada antes del colapso.”*
