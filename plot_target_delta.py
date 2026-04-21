import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_target_delta():
    print("Iniciando Simulador de Target Delta vs Target Absoluto...")
    out_dir = Path("./output")
    subdirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and "gdelt_cache" not in str(d)])
    
    if not subdirs:
        print("[!] No hay corridas previas.")
        return
        
    latest_run = subdirs[-1]
    signals_files = list(latest_run.glob("signals_*.csv"))
    
    if not signals_files:
        print(f"[!] No se encontraron archivos CSV en {latest_run}")
        return
        
    signals_df = pd.read_csv(signals_files[0], index_col=0, parse_dates=True)
    
    # Buscar una ventana de tiempo interesante donde el VIX cruzó 30
    window_df = signals_df[(signals_df["vix"] > 10) & (signals_df["vix"] < 80)].copy()
    if len(window_df) > 200:
        # Extraer muestra alrededor del crash de COVID-19 o VIX > 30 representativo
        crash_dates = window_df[window_df["vix"] > 35].index
        if len(crash_dates) > 0:
            center_date = crash_dates[-1] # tomar un crash reciente
            start_date = center_date - pd.Timedelta(days=100)
            end_date = center_date + pd.Timedelta(days=50)
            vix = window_df.loc[start_date:end_date, "vix"]
        else:
            vix = window_df["vix"].tail(150)
    else:
        vix = signals_df["vix"]

    delta_vix = vix.shift(-1) - vix
    
    # Marcar los días de "Ignition" (Ruptura inicial > 30 desde < 25)
    rupturas = vix[(vix.shift(-1) > 30) & (vix <= 25)]
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("PREDICCIÓN DE ACELERACIÓN vs. PREDICCIÓN DE POSICIÓN (VIX)", fontsize=16, fontweight="bold")
    
    # Panel 1: VIX Absoluto
    ax1 = axes[0]
    ax1.plot(vix.index, vix, color="#FFAA00", lw=2, label="VIX (Posición Absoluta)")
    ax1.axhline(30, color="red", linestyle="--", alpha=0.5, label="Umbral Extremo (>30)")
    ax1.axhline(25, color="white", linestyle=":", alpha=0.3, label="Límite Calma (<25)")
    ax1.set_ylabel("Nivel VIX")
    ax1.set_title("Target Antiguo (Perezoso): Altísima autocorrelación de inercia. Ahoga las variables estructurales.", fontsize=11, color="#888888")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.1)
    
    # Panel 2: VIX Delta
    ax2 = axes[1]
    ax2.plot(delta_vix.index, delta_vix, color="#00ffcc", lw=2, label="Δ VIX (Aceleración Diaria)")
    ax2.axhline(0, color="white", linestyle="-", alpha=0.3)
    ax2.set_ylabel("Cambio en VIX (Puntos)")
    ax2.set_title("Target Nuevo (Delta Causal): Exige predecir la velocidad de ruptura. Extrae el verdadero coeficiente planetario.", fontsize=11, color="#888888")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.1)
    
    if not rupturas.empty:
        for i, ru in enumerate(rupturas.index):
            if ru in delta_vix.index:
                ax1.axvline(ru, color="#FF3366", linestyle="-", lw=2, alpha=0.8, 
                            label="Flash Ignition" if i == 0 else "")
                ax2.scatter(ru, delta_vix[ru], color="#FF3366", s=100, zorder=5, 
                            label="Explosión Delta Inicial" if i == 0 else "")
    
    # Avoid duplicate labels
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc="upper left")
    
    handles2, labels2 = ax2.get_legend_handles_labels()
    by_label2 = dict(zip(labels2, handles2))
    ax2.legend(by_label2.values(), by_label2.keys(), loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = latest_run / "target_delta_simulator.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n[OK] Simulación Target Delta guardada en: {plot_path}")
    
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_target_delta()
