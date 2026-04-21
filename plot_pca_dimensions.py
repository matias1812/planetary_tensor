import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def plot_topological_dimensions():
    # 1. Cargar el último tensor y las señales
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
    
    # Filtrar sólo columnas de tensión planetaria (ignorar global, etc)
    tension_cols = [c for c in tensor_df.columns if c.endswith("_tension")]
    X = tensor_df[tension_cols].dropna()
    
    # Alinear
    df = X.join(signals_df[["vix"]]).dropna()
    X = df[tension_cols]
    
    # 2. Análisis de Componentes Principales (PCA)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    df["Biologico (PCA1)"] = X_pca[:, 0]
    df["Friccion (PCA2)"] = X_pca[:, 1]
    df["Macro Gravedad (PCA3)"] = X_pca[:, 2]
    
    explained_var = pca.explained_variance_ratio_ * 100
    
    # 3. Widget Topológico (Subplots)
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("WIDGET TOPOLÓGICO: LAS 3 DIMENSIONES OCULTAS DE LA MÁQUINA", fontsize=18, fontweight="bold", y=0.95)
    
    # PC1 vs PC3 (Biological vs Industrial)
    ax1 = fig.add_subplot(121)
    scatter1 = ax1.scatter(df["Biologico (PCA1)"], df["Macro Gravedad (PCA3)"], 
                          c=df["vix"], cmap="magma", alpha=0.6, s=15)
    ax1.set_xlabel(f"D1: Metrónomo Biológico (Alta Freq) [{explained_var[0]:.1f}% var]")
    ax1.set_ylabel(f"D3: Gravedad Industrial (Baja Freq) [{explained_var[2]:.1f}% var]")
    ax1.set_title("Topología del Pánico: Cortisol vs Escasez Física")
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label("VIX (Nivel de Pánico)")
    
    # PC2 Time Series (Logistical Friction)
    ax2 = fig.add_subplot(222)
    ax2.plot(df.index[-1000:], df["Friccion (PCA2)"].tail(1000), color="#00ffcc", lw=1)
    ax2.set_title("D2: Fricción Logística (Ping de Redes) - Últimos 1000 días")
    ax2.set_ylabel("Tensión de Redes (PCA)")
    ax2.grid(True, alpha=0.1)
    
    # Cargas de los Componentes (Feature Weights)
    ax3 = fig.add_subplot(224)
    components = pd.DataFrame(pca.components_, columns=tension_cols, index=["D1_Bio", "D2_Net", "D3_Macro"])
    top_features = components.abs().max().sort_values(ascending=False).head(10).index
    sns.heatmap(components[top_features].T, cmap="coolwarm", center=0, ax=ax3, cbar=False)
    ax3.set_title("Pesos Planetarios en el ADN del PCA")
    
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(latest_run / "topological_dimensions_pca.png", dpi=150)
    print(f"\n[OK] Simulación Topológica guardada en: {latest_run}/topological_dimensions_pca.png")
    
    # Opcional: mostrar gráfico si no estamos en entorno headless
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    plot_topological_dimensions()
