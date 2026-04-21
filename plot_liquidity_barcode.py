import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_liquidity_barcode():
    print("Iniciando Simulador del Código de Barras de Liquidez (D2)...")
    out_dir = Path("./output")
    subdirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and "gdelt_cache" not in str(d)])
    
    if not subdirs:
        print("[!] No hay corridas previas. Ejecuta planetary_tensor_analysis.py primero.")
        return
        
    latest_run = subdirs[-1]
    tensor_files = list(latest_run.glob("tensor_*.csv"))
    signals_files = list(latest_run.glob("signals_*.csv"))
    
    if not tensor_files or not signals_files:
        print(f"[!] No se encontraron archivos CSV en {latest_run}")
        return
        
    tensor_df = pd.read_csv(tensor_files[0], index_col=0, parse_dates=True)
    signals_df = pd.read_csv(signals_files[0], index_col=0, parse_dates=True)
    
    tension_cols = [c for c in tensor_df.columns if c.endswith("_tension")]
    X = tensor_df[tension_cols].dropna()
    
    df = X.join(signals_df[["vix"]]).dropna()
    X = df[tension_cols]
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    df["D2_Liquidity"] = X_pca[:, 1] # Fricción / Logística
    
    # Enfocarse en los últimos 2000 días para que el "Código de Barras" sea visible
    window_df = df.tail(2000)
    
    # Identificar Shocks de Liquidez (D2 cae por debajo de su mediana inferior, ej. -0.5)
    liquidity_shocks = window_df[window_df["D2_Liquidity"] < -0.5]
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1, 3]})
    fig.suptitle("CÓDIGO DE BARRAS DE LIQUIDEZ: APAGONES INSTITUCIONALES (DIMENSIÓN D2)", fontsize=16, fontweight="bold")
    
    # Panel 1: Código de Barras Binario (Shock vs Normal)
    ax1 = axes[0]
    ax1.fill_between(window_df.index, 0, 1, where=(window_df["D2_Liquidity"] < -0.5), 
                     color="#FF3366", alpha=0.8, step="pre", label="Shock de Liquidez (Congelamiento Macroeconómico)")
    ax1.set_yticks([])
    ax1.set_xlim(window_df.index[0], window_df.index[-1])
    ax1.set_title("Onda Cuadrada: ON (Flujo Continuo) / OFF (Shock Pivot / Parada Súbita)", color="#00ffcc", fontsize=11)
    ax1.legend(loc="upper left")
    
    # Panel 2: Serie de Tiempo Cruda
    ax2 = axes[1]
    ax2.plot(window_df.index, window_df["D2_Liquidity"], color="#00ffcc", lw=1.5, label="Fricción Logística D2 (Júpiter-Urano-Mercurio)")
    
    # Marcar los Shocks
    ax2.scatter(liquidity_shocks.index, liquidity_shocks["D2_Liquidity"], color="#FF3366", s=40, zorder=5)
    ax2.axhline(0, color="white", lw=1, ls="--", alpha=0.3)
    ax2.axhline(-0.5, color="#FF3366", lw=1, ls=":", label="Umbral Crítico (-0.5)")
    
    ax2.set_ylabel("Fuerza Relativa (PCA Vector)")
    ax2.set_xlim(window_df.index[0], window_df.index[-1])
    ax2.grid(True, alpha=0.15)
    ax2.legend(loc="upper right")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = latest_run / "liquidity_barcode_simulator.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Código de Barras Logístico guardado en: {plot_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_liquidity_barcode()
