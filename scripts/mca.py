from prince import MCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from itertools import combinations


def plot_scree_mca(mca):
    """
    Dibuja el gráfico de sedimentación (Scree Plot) para MCA.

    Args:
        mca (prince.MCA): Modelo de MCA entrenado.
    """
    eig_vals = mca.eigenvalues_
    var_exp = eig_vals / eig_vals.sum()  # Varianza explicada

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(eig_vals) + 1), var_exp.cumsum(), marker='o', linestyle='--')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Varianza Acumulada')
    plt.title('Scree Plot - MCA')
    plt.grid()
    plt.show()


def apply_mca(df, variables, target_var=None, n_components=2):
    """
    Aplica el Análisis de Correspondencias Múltiples (MCA) a un conjunto de datos.
    
    Args:
        df (DataFrame): Conjunto de datos original.
        variables (list): Lista de variables a incluir en el análisis.
        n_components (int): Número de componentes principales a extraer.

    Returns:
        mca (prince.MCA): Modelo de MCA entrenado.
        df_mca_coords (DataFrame): Coordenadas de observaciones en el espacio MCA con variables originales.
    """
    # # Filtrar las columnas categóricas y eliminar valores faltantes
    df_train = df[variables]
    if target_var:
        df_target = df[target_var]

    # Aplicar MCA
    mca = MCA(n_components=n_components, engine='sklearn', random_state=42)
    mca.fit(df_train)

    # Obtener coordenadas de las filas
    df_mca_coords = mca.row_coordinates(df_train)
    df_mca_coords.columns = [f'Dim{i+1}' for i in range(n_components)]  # Renombrar columnas

    # Agregar las variables originales al DataFrame de coordenadas
    df_mca_coords = pd.concat([df_mca_coords, df_train.reset_index(drop=True)], axis=1)
    df_mca_coords = pd.concat([df_mca_coords, df_target.reset_index(drop=True)], axis=1)

    # Imprimir varianza explicada
    print("Varianza Explicada por Dimensión:")
    print(mca.eigenvalues_summary)

    return mca, df_mca_coords


def optimize_mca_variance(df, always_include, candidate_variables, n_components=2):
    """
    Optimiza la varianza explicada en las dos primeras dimensiones de MCA,
    probando combinaciones de variables adicionales.

    Args:
        df (DataFrame): Datos originales.
        always_include (list): Lista de variables que deben incluirse siempre.
        candidate_variables (list): Lista de variables opcionales para probar.
        n_components (int): Número de dimensiones principales a calcular.

    Returns:
        dict: Diccionario con:
            - "variables": Lista de combinaciones de variables utilizadas.
            - "varianza": Lista de varianza explicada acumulada en las dos primeras dimensiones.
    """
    results = {"variables": [], "varianza_dim1": [], "varianza_dim2": [], "varianza_total": []}

    # Generar todas las combinaciones de las variables candidatas
    for r in range(len(candidate_variables) + 1):  # De 0 a todas las variables candidatas
        for combo in combinations(candidate_variables, r):
            # Variables a incluir en esta iteración
            selected_variables = always_include + list(combo)
            
            # Filtrar y convertir las variables a categóricas
            df_filtered = df[selected_variables].dropna()
            df_filtered = df_filtered.apply(lambda x: x.astype('category'))

            # Aplicar MCA
            mca = MCA(n_components=n_components, engine='sklearn', random_state=42)
            mca.fit(df_filtered)

            # Obtener varianza explicada acumulada en las primeras dos dimensiones
            explained_variance_1 = mca.eigenvalues_[0]
            explained_variance_2 = mca.eigenvalues_[1]
            explained_variance_sum = explained_variance_1 + explained_variance_2

            # Guardar los resultados
            results["variables"].append(selected_variables)
            results["varianza_dim1"].append(explained_variance_1)
            results["varianza_dim2"].append(explained_variance_2)
            results["varianza_total"].append(explained_variance_sum)

    return results


def plot_mca(
    df_coords,
    x_dim="Dim1",
    y_dim="Dim2",
    hue_var=None,
    style_var=None,
    title="Mapa Factorial - MCA"
):
    """
    Genera un gráfico bidimensional del espacio MCA con opciones para cambiar dimensiones, colores y estilos.
    
    Args:
        df_coords (DataFrame): Coordenadas de observaciones en el espacio MCA con variables originales.
        x_dim (str): Nombre de la dimensión en el eje X (por defecto, 'Dim1').
        y_dim (str): Nombre de la dimensión en el eje Y (por defecto, 'Dim2').
        hue_var (str, opcional): Variable categórica para colorear los puntos.
        style_var (str, opcional): Variable categórica para diferenciar los puntos por estilo/forma.
        title (str): Título del gráfico.
    """
    plt.figure(figsize=(12, 8))

    # Validar que las columnas x_dim y y_dim existen
    if x_dim not in df_coords.columns or y_dim not in df_coords.columns:
        raise ValueError(f"El DataFrame no contiene las columnas {x_dim} y/o {y_dim} necesarias para graficar.")
    
    sns.scatterplot(
        data=df_coords,
        x=x_dim,
        y=y_dim,
        hue=hue_var,
        style=style_var,
        alpha=0.7,
        palette="Set2"
    )

    # Ajustar leyendas y etiquetas
    if hue_var or style_var:
        legend_title = f"{hue_var}" + (f" / {style_var}" if style_var else "")
        plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")
    
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.title(title)
    plt.xlabel(x_dim)
    plt.ylabel(y_dim)
    plt.show()


def plot_multiple_mca(df_coords, configs, figsize=(16, 12)):
    """
    Genera múltiples gráficos MCA en una misma figura.
    
    Args:
        df_coords (DataFrame): Coordenadas del MCA con variables originales.
        configs (list of dict): Lista de configuraciones para los gráficos. Cada configuración debe ser un dict con:
            - x_dim (str): Dimensión para el eje X.
            - y_dim (str): Dimensión para el eje Y.
            - hue_var (str): Variable para colorear los puntos.
            - style_var (str, opcional): Variable para diferenciar por forma.
            - title (str, opcional): Título del subgráfico.
        figsize (tuple): Tamaño de la figura total.
    """
    n_plots = len(configs)
    n_cols = 2
    n_rows = (n_plots + 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Aplanar para facilitar el acceso a los ejes

    for i, config in enumerate(configs):
        ax = axes[i]
        sns.scatterplot(
            data=df_coords,
            x=config.get("x_dim", "Dim1"),
            y=config.get("y_dim", "Dim2"),
            hue=config.get("hue_var"),
            style=config.get("style_var"),
            alpha=0.7,
            palette="Set2",
            ax=ax
        )
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(config.get("title", "Mapa Factorial"))
        ax.set_xlabel(config.get("x_dim", "Dim1"))
        ax.set_ylabel(config.get("y_dim", "Dim2"))
        if config.get("hue_var") or config.get("style_var"):
            ax.legend(title=config.get("hue_var") + (f" / {config.get('style_var')}" if config.get("style_var") else ""))
    
    # Eliminar ejes vacíos si hay menos gráficos que subgráficos
    for j in range(len(configs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Función para graficar en el espacio MCA
def plot_mca_old(df_coords, x_var, y_var=None, title='Mapa Factorial - MCA'):
    plt.figure(figsize=(12, 8))
    if y_var:
        sns.scatterplot(data=df_coords, x='Dim1', y='Dim2', hue=x_var, style=y_var, alpha=0.7, palette='Set2')
        plt.legend(title=f'{x_var} / {y_var}', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        sns.scatterplot(data=df_coords, x='Dim1', y='Dim2', hue=x_var, alpha=0.7, palette='Set2')
        plt.legend(title=x_var, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.xlabel('Dimensión 1')
    plt.ylabel('Dimensión 2')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.show()
