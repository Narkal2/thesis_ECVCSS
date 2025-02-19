import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

def generar_tabla_frecuencias(df, variable):
    """
    Genera una tabla con las frecuencias absolutas y porcentajes de una variable específica.

    Parámetros:
    - df: DataFrame con los datos.
    - variable: Nombre de la variable para calcular las frecuencias.

    Retorna:
    - tabla_frecuencias: DataFrame con las frecuencias absolutas y porcentajes.
    """
    # Calcular las frecuencias absolutas
    frecuencias = df[variable].value_counts().reset_index()
    frecuencias.columns = [variable, 'Personas usuarias']
    
    # Calcular los porcentajes
    frecuencias['Porcentaje'] = (frecuencias['Personas usuarias'] / 
                                 frecuencias['Personas usuarias'].sum() * 100).round(1).astype(float)
    
    # Agregar fila de totales
    total_row = pd.DataFrame({
        variable: ['Total'],
        'Personas usuarias': [frecuencias['Personas usuarias'].sum()],
        'Porcentaje': [100.0]
    })
    
    tabla_frecuencias = pd.concat([frecuencias, total_row], ignore_index=True)

    # Ajustar formato para la salida
    pd.options.display.float_format = '{:.1f}'.format
    
    return tabla_frecuencias

# Ejemplo de uso:
# tabla = generar_tabla_frecuencias(df, 'Tipo_Hogar')
# print(tabla)


def analizar_cruce_bivariable(df, variable_fila, variable_columna):
    """
    Genera una tabla cruzada con frecuencias absolutas y relativas, y calcula
    los estadísticos Chi-cuadrado (Χ²) y Phi entre dos variables categóricas.

    Parámetros:
    - df: DataFrame con los datos.
    - variable_fila: Nombre de la variable para las filas.
    - variable_columna: Nombre de la variable para las columnas.

    Retorna:
    - tabla_combinada: DataFrame con frecuencias absolutas y relativas.
    - chi2: Valor Chi-cuadrado.
    - phi: Coeficiente Phi.
    - cramer_v: Coeficiente de asociación de Cramér's V.
    - p: Valor p asociado al Chi-cuadrado.
    """
    # Crear tabla cruzada
    tabla_cruzada = pd.crosstab(df[variable_fila], df[variable_columna], margins=True, margins_name="Total")
    
    # Calcular frecuencias relativas
    tabla_cruzada_pct = tabla_cruzada.div(tabla_cruzada.loc["Total"], axis=1) * 100
    
    # Combinar frecuencias absolutas y relativas
    tabla_combinada = pd.DataFrame()
    for col in tabla_cruzada.columns:
        tabla_combinada[(col, 'Casos')] = tabla_cruzada[col]
        tabla_combinada[(col, 'Porcentaje')] = tabla_cruzada_pct[col].round(1).astype(float)
    
    # Renombrar columnas para claridad
    tabla_combinada.columns = pd.MultiIndex.from_tuples(tabla_combinada.columns)
    tabla_combinada.index.name = variable_fila

    # Ajustar formato para la salida
    pd.options.display.float_format = '{:.1f}'.format

    # Cálculo de Chi-cuadrado y Phi
    chi2, p, dof, expected = chi2_contingency(tabla_cruzada.iloc[:-1, :-1])  # Excluir la fila y columna "Total"
    n_total = tabla_cruzada.iloc[:-1, :-1].sum().sum()  # Total de observaciones

    phi = np.sqrt(chi2 / n_total)

    # Determinar el tamaño de la tabla
    num_filas, num_columnas = tabla_cruzada.shape[0] - 1, tabla_cruzada.shape[1] - 1  # Excluir 'Total'

    # Calcular Cramér's V
    min_dim = min(num_filas, num_columnas)
    cramer_v = np.sqrt(chi2 / (n_total * (min_dim - 1))) if min_dim > 1 else np.sqrt(chi2 / n_total)

    return tabla_combinada, chi2, phi, cramer_v, p

# Ejemplo de uso:
# tabla, chi2, phi, cramer_v, p = analizar_cruce_bivariable(df, 'Pais_Nacimiento', 'Edad_Recod')
# print(tabla)
# print(f"Chi-cuadrado (Χ²): {chi2:.2f}")
# print(f"Coeficiente Phi: {phi:.3f}")
# print(f"Valor p: {p:.4f}")

def plot_table(table, columnas_casos=[]):
    """
    Muestra una tabla con líneas y formato visual en un Jupyter Notebook.

    Parámetros:
    - tabla: DataFrame a mostrar con estilo.
    """
    from IPython.display import display
    
    # Aplicar estilo
    styled_table = table.style.set_table_styles([
        {'selector': 'thead th', 'props': [('border', '1px solid black'), ('text-align', 'center')]},
        {'selector': 'tbody td', 'props': [('border', '1px solid black'), ('text-align', 'center')]}
    ]).set_properties(**{'text-align': 'center'}).format(precision=1)

    # Aplicar degradado a las columnas especificadas
    if columnas_casos:
        for columna in columnas_casos:
            styled_table = styled_table.background_gradient(subset=[columna], cmap='Blues')

    # Mostrar tabla
    display(styled_table)

# Ejemplo de uso:
# table = generar_tabla_frecuencias(df, 'Tipo_Hogar')
# plot_table(table)
