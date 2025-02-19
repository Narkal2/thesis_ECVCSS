import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Configuración de estilo para los gráficos
sns.set_theme(style="whitegrid")


def plot_bar(df, x, hue=None, title="Gráfico de Barras"):
    """
    Genera un gráfico de barras para una variable categórica.
    Parámetros:
        - df: DataFrame de pandas
        - x: Variable categórica a graficar en el eje x
        - hue: Variable opcional para agrupar por colores
        - title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))

    if hue:
        sns.countplot(data=df, x=x, hue=hue, palette="viridis")
    else:
        sns.countplot(data=df, x=x, hue=x, palette="viridis", legend=False)

    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()


def plot_box(df, x, y, y_label=None, hue=None, palette="Set2", title="Gráfico de Caja"):
    """
    Genera un boxplot para analizar la distribución de una variable numérica según una categórica.
    Parámetros:
        - df: DataFrame de pandas
        - x: Variable categórica
        - y: Variable numérica
        - hue: Variable opcional para agrupar por colores
        - title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))

    if hue:
        sns.boxplot(data=df, x=x, y=y, hue=hue, palette="coolwarm")
    else:
        sns.boxplot(data=df, x=x, y=y, palette=sns.color_palette(palette))

    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel(x.replace("_", " "))
    plt.ylabel(y_label if y_label else y.replace("_", " "))  # Usa el nombre modificado si está definido
    plt.show()


def plot_violin(df, x, y, hue=None, title="Gráfico de Violín"):
    """
    Genera un gráfico de violín para visualizar la distribución de una variable numérica según una categórica.
    Parámetros:
        - df: DataFrame de pandas
        - x: Variable categórica
        - y: Variable numérica
        - hue: Variable opcional para agrupar por colores
        - title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))

    if hue:
        sns.violinplot(data=df, x=x, y=y, hue=hue, palette="muted", split=True)
    else:
        sns.violinplot(data=df, x=x, y=y, palette="muted")

    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()


def plot_correlation_matrix(df, title="Matriz de Correlación", annot_size=8, rename_dict=None, cmap="coolwarm", center_value=None):
    """
    Genera un mapa de calor con la matriz de correlación entre variables numéricas.

    Parámetros:
        - df: DataFrame de pandas con las variables numéricas.
        - title: Título del gráfico.
        - annot_size: Tamaño del texto de los valores de correlación.
        - rename_dict: Diccionario opcional para cambiar los nombres de las variables en el plot.
        - cmap: Paleta de colores para el heatmap.
        - center_value: Valor en el que se centra la escala de colores.
    """
    plt.figure(figsize=(10, 8))
    
    # Calcular la matriz de correlación
    correlation = df.corr()

    # Renombrar las variables en el eje X e Y
    if rename_dict:
        correlation = correlation.rename(index=rename_dict, columns=rename_dict)

    # Crear el heatmap con la nueva configuración
    sns.heatmap(
        correlation, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap, 
        center=center_value,  # Permite centrar la escala en un valor específico
        linewidths=0.5, 
        annot_kws={"size": annot_size}  # Ajuste del tamaño del texto
    )

    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()


def plot_scatter(df, x, y, hue=None, title="Gráfico de Dispersión"):
    """
    Genera un scatter plot para analizar relaciones entre variables numéricas.
    Parámetros:
        - df: DataFrame de pandas
        - x: Variable numérica en eje x
        - y: Variable numérica en eje y
        - hue: Variable opcional para colorear por categoría
        - title: Título del gráfico
    """
    plt.figure(figsize=(10, 6))

    if hue:
        sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="deep")
    else:
        sns.scatterplot(data=df, x=x, y=y, palette="deep")

    plt.title(title)
    plt.show()


def plot_multiple_numeric(df, numeric_vars, category, plot_type="boxplot", title="Comparación de Valoraciones"):
    """
    Genera una única gráfica con múltiples variables numéricas separadas por una variable categórica.
    
    Parámetros:
        - df: DataFrame de pandas
        - numeric_vars: Lista con los nombres de las variables numéricas a comparar.
        - category: Variable categórica para separar los datos (e.g., "Sexo").
        - plot_type: Tipo de gráfico a usar ("boxplot", "violin", "barplot").
        - title: Título del gráfico.
    """
    # Transformar los datos de wide format a long format para Seaborn
    df_long = df.melt(id_vars=[category], value_vars=numeric_vars, var_name="Variable", value_name="Valoración")

    plt.figure(figsize=(12, 6))

    if plot_type == "boxplot":
        sns.boxplot(data=df_long, x="Variable", y="Valoración", hue=category, palette="coolwarm")
    elif plot_type == "violin":
        sns.violinplot(data=df_long, x="Variable", y="Valoración", hue=category, palette="muted", split=True)
    elif plot_type == "barplot":
        sns.barplot(data=df_long, x="Variable", y="Valoración", hue=category, palette="viridis", ci="sd")
    else:
        raise ValueError("Tipo de gráfico no válido. Usa 'boxplot', 'violin' o 'barplot'.")

    plt.xticks(rotation=45)
    plt.title(title)
    plt.legend(title=category)
    plt.show()


def plot_stacked_bar_percentage(df, category_x, category_hue, title="Gráfico de Barras Apiladas (%)",
                                rename_dict_vars=None, rename_dict_values=None, 
                                custom_order_dict=None, order_by_hue_value=None,
                                cmap="Pastel2"):
    """
    Genera un gráfico de barras apiladas basado en porcentajes con opciones avanzadas de personalización.

    Parámetros:
        - df: DataFrame de pandas con los datos.
        - category_x: Variable categórica que estará en el eje X.
        - category_hue: Variable categórica que segmentará los valores en la leyenda.
        - title: Título del gráfico.
        - rename_dict_vars: Diccionario opcional para cambiar el nombre de las variables en los ejes.
        - rename_dict_values: Diccionario opcional para cambiar los nombres de las clases de cada variable.
        - custom_order_dict: Diccionario opcional para ordenar manualmente las categorías del eje X.
        - order_by_hue_value: Si se proporciona, ordena el eje X en función del porcentaje de un valor específico en `category_hue`.
    """
    # Crear una tabla de contingencia y calcular los porcentajes
    crosstab = pd.crosstab(df[category_x], df[category_hue], normalize='index') * 100

    # Renombrar valores dentro de la tabla de contingencia si se proporciona un diccionario
    if rename_dict_values:
        crosstab.columns = [rename_dict_values.get(col, col) for col in crosstab.columns]
        crosstab.index = [rename_dict_values.get(idx, idx) for idx in crosstab.index]

    # Ordenar las categorías en el eje X
    if custom_order_dict:
        order = sorted(crosstab.index, key=lambda x: custom_order_dict.get(x, float('inf')))
    elif order_by_hue_value:
        if order_by_hue_value in crosstab.columns:
            order = crosstab[order_by_hue_value].sort_values(ascending=False).index
        else:
            print(f"Advertencia: '{order_by_hue_value}' no está en la variable {category_hue}. Se usará orden original.")
            order = crosstab.index
    else:
        order = crosstab.index

    # Renombrar la variable del eje X y la leyenda si se proporciona un diccionario
    x_label = rename_dict_vars.get(category_x, category_x).replace("_", " ") if rename_dict_vars else category_x.replace("_", " ")
    hue_label = rename_dict_vars.get(category_hue, category_hue).replace("_", " ") if rename_dict_vars else category_hue.replace("_", " ")

    # Crear el gráfico
    plt.figure(figsize=(12, 6))
    ax = crosstab.loc[order].plot(kind='bar', stacked=True, colormap=cmap, figsize=(12, 6), edgecolor="black", linewidth=0.5)

    # Personalización del gráfico
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Porcentaje (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title=hue_label, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()


def plot_stacked_bar_percentage_binary(df, category_vars, rename_dict=None, x_label="Variables", title="Distribución de Accesos y Dificultades (%)"):
    """
    Genera un gráfico de barras apiladas basado en porcentajes para múltiples variables binarias.

    Parámetros:
        - df: DataFrame de pandas con los datos.
        - category_vars: Lista de variables binarias a graficar.
        - rename_dict: Diccionario opcional para renombrar las variables en el gráfico.
        - title: Título del gráfico.
    """
    # Transformar los datos a formato largo (long format)
    df_long = df[category_vars].melt(var_name="Variable", value_name="Respuesta")

    # Renombrar las variables si se proporciona un diccionario
    if rename_dict:
        df_long["Variable"] = df_long["Variable"].replace(rename_dict)

    # Crear una tabla de frecuencias en porcentaje
    crosstab = pd.crosstab(df_long["Variable"], df_long["Respuesta"], normalize="index") * 100

    # Ordenar de mayor a menor según la frecuencia de "Sí"
    order = crosstab["Sí"].sort_values(ascending=False).index

    # Graficar
    plt.figure(figsize=(18, 10))
    ax = crosstab.loc[order].plot(kind="bar", stacked=True, colormap="Pastel1", figsize=(12, 6), edgecolor="black", linewidth=0.5)

    # Personalización del gráfico
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Porcentaje (%)")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Respuesta", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Mostrar el gráfico
    plt.show()

# Ejemplo de uso:

# Uso de la función con el DataFrame
# plot_stacked_bar_percentage_binary(df_filtrado, variables_acceso, rename_dict, x_label="Acceso a bienes y servicios", title="Distribución (%) para las personas mayores")

# Ejemplo de uso:
# plot_bar(df, "Tipo_Hogar")
# plot_box(df, "Estado_Civil", "Edad", hue="Sexo")
# plot_violin(df, "Nivel_Estudios", "Edad", hue="Sexo")
# plot_correlation_matrix(df)
# plot_scatter(df, "Edad", "Ingresos", hue="Sexo")


# Ejemplo de uso:
# plot_multiple_numeric(df, ["Valoracion_General", "Valoracion_Utilidad", "Valoracion_Utilidad2"], "Sexo", plot_type="boxplot")
# plot_multiple_numeric(df, ["Valoracion_General", "Valoracion_Utilidad", "Valoracion_Utilidad2"], "Sexo", plot_type="violin")
# plot_multiple_numeric(df, ["Valoracion_General", "Valoracion_Utilidad", "Valoracion_Utilidad2"], "Sexo", plot_type="barplot")


# Ejemplo de uso:
# plot_stacked_bar_percentage(df, "Nivel_Estudios", "Edad_Recod2", title="Distribución de Edad por Nivel de Estudios")
