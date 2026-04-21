import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_data_alignment_leakage():
    print("Iniciando Simulador Cuantitativo de Fuga de Datos...")
    # Cargar último run
    out_dir = Path("./output")
    subdirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and "gdelt_cache" not in str(d)])
    
    if not subdirs:
        print("[!] No hay corridas previas.")
        return
        
    latest_run = subdirs[-1]
    tensor_files = list(latest_run.glob("tensor_*.csv"))
    signals_files = list(latest_run.glob("signals_*.csv"))
    
    if not tensor_files or not signals_files:
        print("[!] CSVs no encontrados.")
        return
        
    tensor_df = pd.read_csv(tensor_files[0], index_col=0, parse_dates=True)
    signals_df = pd.read_csv(signals_files[0], index_col=0, parse_dates=True)
    
    # 1. Simular la matriz CON Data Leakage (Alineación contemporánea)
    df_leak = pd.DataFrame()
    df_leak["Hoy: Tensión Saturno-Urano"] = tensor_df["saturn_uranus_tension"]
    df_leak["TARGET FALSO: VIX de Hoy (El algoritmo hace trampa)"] = signals_df["vix"]
    df_leak = df_leak.dropna().tail(100) # Evaluar los últimos 100 días
    
    # 2. Simular la matriz SIN Data Leakage (Alineación Predictiva Real OOS)
    df_real = pd.DataFrame()
    df_real["Hoy: Tensión Saturno-Urano"] = tensor_df["saturn_uranus_tension"]
    df_real["TARGET REAL: VIX de Mañana (shift(-1))"] = signals_df["vix"].shift(-1)
    df_real = df_real.dropna().tail(100)
    
    # Graficar y Visualizar la Trampa vs La Realidad
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle("Simulador de Data Leakage: Anatomía del Bug de 'Visión Divina'", fontsize=16, fontweight="bold")
    
    # Plot Trampa
    ax1 = axes[0]
    ax1.plot(df_leak.index, df_leak["Hoy: Tensión Saturno-Urano"], color="#7F77DD", lw=2, label="Feature: Tensión de HOY")
    ax1.plot(df_leak.index, df_leak["TARGET FALSO: VIX de Hoy (El algoritmo hace trampa)"], color="#E24B4A", lw=2, label="Target: VIX de HOY")
    ax1.set_title("CON LEAKAGE: El algoritmo aprende a correlacionar el mismo instante (inutilizable en Trading)", fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    
    # Plot Realidad
    ax2 = axes[1]
    ax2.plot(df_real.index, df_real["Hoy: Tensión Saturno-Urano"], color="#7F77DD", lw=2, label="Feature: Tensión de HOY")
    ax2.plot(df_real.index, df_real["TARGET REAL: VIX de Mañana (shift(-1))"], color="#00ffcc", lw=2, ls="--", label="Target: VIX de MAÑANA (shift -1)")
    ax2.set_title("SIN LEAKAGE: Adivinando el precio de mañana con la tensión de hoy (El Verdadero OOS Edge)", fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    for ax in axes:
        ax.set_ylabel("Valor")
        
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = latest_run / "data_leakage_simulator.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Simulación Visual guardada en: {plot_path}")
    
    # Mostrar si es posible
    try:
        plt.show()
    except:
        pass

if __name__ == "__main__":
    plot_data_alignment_leakage()
