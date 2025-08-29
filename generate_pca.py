#!/usr/bin/env python3
"""Generador de datos sintéticos para ejemplos de PCA.

Crea un dataset con estructura de baja dimensión intrínseca (k) mapeada a
un espacio de características de dimensión p mediante una transformación
lineal, y añade ruido gaussiano. Ideal para demostrar PCA.

Salida: CSV en `data/pca_synthetic.csv` (o la ruta indicada).
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_low_rank_data(
    n_samples=200, n_features=20, intrinsic_dim=3, noise=0.1, seed=None
):
    rng = np.random.RandomState(seed)
    # Latent factors Z ~ N(0, diag(sigma^2)) with decreasing variances
    sigmas = np.linspace(1.0, 0.2, intrinsic_dim)
    Z = rng.normal(scale=1.0, size=(n_samples, intrinsic_dim)) * sigmas

    # Random linear map from latent space to observed features
    A = rng.normal(scale=1.0, size=(intrinsic_dim, n_features))

    X_clean = Z.dot(A)
    X_noise = X_clean + rng.normal(scale=noise, size=X_clean.shape)

    return X_noise


def save_dataframe(X, out_path):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "id", range(1, len(df) + 1))
    df.to_csv(out_path, index=False)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generador de datos sintéticos para PCA"
    )
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Número de filas (muestras)"
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=20,
        help="Número de características (dimensión observada)",
    )
    parser.add_argument(
        "--intrinsic-dim",
        type=int,
        default=3,
        help="Dimensión intrínseca (número de factores latentes)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Desviación estándar del ruido Gaussiano",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    parser.add_argument(
        "--out", type=str, default="data/pca_synthetic.csv", help="Ruta de salida CSV"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta una demostración de PCA y muestra explained variance",
    )

    args = parser.parse_args()

    X = generate_low_rank_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        intrinsic_dim=args.intrinsic_dim,
        noise=args.noise,
        seed=args.seed,
    )

    df = save_dataframe(X, args.out)
    print(f"Guardado: {args.out} — {df.shape[0]} filas x {df.shape[1]} columnas")

    if args.demo:
        try:
            from sklearn.decomposition import PCA

            pca = PCA()
            vals = pca.fit_transform(X)
            evr = pca.explained_variance_ratio_
            print("Explained variance ratio (primeras 10 componentes):")
            print(np.round(evr[:10], 4))
            cum = np.cumsum(evr)
            print("Varianza acumulada (primeras 10):", np.round(cum[:10], 4))
        except Exception as e:
            print("Demo de PCA falló — asegúrate de tener scikit-learn instalado:", e)


if __name__ == "__main__":
    main()
