import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def filter_variables(df):
    """
    Filtra el DataFrame según los siguientes criterios:
    
    1. Para las variables categóricas (tipo 'object' o 'category'),
       elimina las observaciones que contengan: "No ho sap", "No contesta" o "NS/NC".
       
    2. Para las variables numéricas que tengan en el nombre "Valoracion" o que
       estén en la lista:
       ['Millora_Autonomia', 'Millora_Anim', 'Sentimient_Companyia', 
        'Sentimient_Tranquilitat', 'Sentimient_Seguretat', 'Satisfaccio_Vital', 
        'Benefici_Familia'],
       elimina las observaciones cuyos valores sean exactamente 98 o 99.
       
    Args:
        df (DataFrame): DataFrame de entrada con las variables preseleccionadas.
        
    Returns:
        DataFrame: Una copia del DataFrame filtrado según los criterios anteriores.
    """
    # Imprimir shape original
    print("Shape original del DataFrame:", df.shape)
    
    # Crear una copia para no modificar el DataFrame original
    df_filtered = df.copy()
    df_filtered = df_filtered.dropna()
    
    # 1. Filtrar variables categóricas
    invalid_cat = {"No ho sap", "No contesta", "NS/NC"}
    # Seleccionar columnas categóricas utilizando select_dtypes
    categorical_columns = df_filtered.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        # Eliminar las filas que tengan valores no válidos en la columna
        df_filtered = df_filtered[~df_filtered[col].isin(invalid_cat)]
    
    # 2. Filtrar variables numéricas específicas
    additional_list = [
        'Millora_Autonomia', 'Millora_Anim', 'Sentimient_Companyia', 
        'Sentimient_Tranquilitat', 'Sentimient_Seguretat', 'Satisfaccio_Vital', 
        'Benefici_Familia'
    ]
    invalid_num = {98, 99}
    # Seleccionar columnas numéricas (int y float)
    numerical_columns = df_filtered.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_columns:
        # Si el nombre de la columna contiene "Valoracion" o está en additional_list, filtrar
        if ("Valoracion" in col) or (col in additional_list):
            df_filtered = df_filtered[~df_filtered[col].isin(invalid_num)]
    
    # Imprimir shape después del filtrado
    print("Shape del DataFrame después del filtrado:", df_filtered.shape)
    
    return df_filtered

# Ejemplo de uso:
#variables_preseleccionadas = ['Categoria_Profesional', 'Valoracion_General']
#df_filtered = filter_variables(df[variables_preseleccionadas])


# !! En MCA no hace falta utilizarlo, porque prince.MCA ya se encarga de codificar las variables categóricas
def process_categorical_data(df, variables, drop_first=False):
    """
    Preprocesa las variables categóricas para el análisis de componentes principales.
    
    Args:
        df (DataFrame): Conjunto de datos original.
        variables (list): Lista de variables categóricas a incluir en el análisis.
        
    Returns:
        ndarray: Datos preprocesados y listos para el análisis.
    """
    df_filtered = df[variables].dropna()

    if drop_first:
        encoder = OneHotEncoder(drop='first')
    else:
        encoder = OneHotEncoder()
    
    encoded_data = encoder.fit_transform(df_filtered).toarray()
    encoded_feature_names = encoder.get_feature_names_out(variables)

    return encoded_data, encoded_feature_names