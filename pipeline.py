import polars as pl
from rapidfuzz import process, fuzz
from collections import Counter

# Rellenar valores nulos con una fecha ficticia
def fill_date_null(df, dates, date_fict):
    for col in dates:
        df = df.with_columns(
            pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False).fill_null(date_fict).alias(col)
        )
    return df

# Preprocesar nombres de compañías con expresiones vectorizadas
def preprocess_company(df, column):
    return df.with_columns(
        pl.col(column)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r'\b(inc|corp|corporation|limited|ltd|plc|pictures|picture|co|entertainment|films|film|studios|corporat)\b', '')
        .str.replace_all(r'[^\w\s]', '')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .alias(column)
    )

# Normalización con fuzzy matching
def group_similar(company, all_companies, threshold=85):
    matches = process.extract(company, all_companies, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if matches:
        match_names = [match[0] for match in matches]
        most_common = Counter(match_names).most_common(1)
        return most_common[0][0] if most_common else company
    return company

# Cargar datos en un LazyFrame
original_data = pl.scan_csv(r'data/rotten_tomatoes_movies.csv')

# Eliminar columnas irrelevantes y filas con valores faltantes
raw_data = (original_data
    .drop(['rotten_tomatoes_link', 'movie_info', 'critics_consensus', 'tomatometer_count',
            'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count',
            'tomatometer_rotten_critics_count', 'audience_count'])
    .drop_nulls(['movie_title'])
)

# Eliminar duplicados
raw_data = raw_data.unique(subset=['movie_title', 'original_release_date', 'directors'])

# Verificar los datos después de cargar
print("Datos después de la carga y limpieza:")
print(raw_data.select(['movie_title', 'production_company']).head(10).collect())

# Rellenar fechas ficticias
dates = ['original_release_date', 'streaming_release_date']
raw_data = fill_date_null(raw_data, dates, '1900-01-01')

# Normalizar los nombres de las compañías
raw_data = preprocess_company(raw_data, 'production_company')

# Verificar los datos después de la normalización
print("Datos después de la normalización de compañías:")
print(raw_data.select(['production_company']).head(10).collect())

# Obtener la lista de compañías para usar en la agrupación
companies_list = raw_data.select('production_company').collect().to_series().drop_nulls().to_list()
print("Lista de compañías (sin nulos):", companies_list)

# Agrupar compañías similares utilizando la lista de compañías
raw_data = raw_data.with_columns(
    pl.col('production_company').map_elements(
        lambda company: group_similar([company], companies_list, threshold=85)[0] if company else 'nr',
        return_dtype=pl.Utf8
    ).alias('production_company')
)

# Ejecutar las operaciones y mostrar resultados
final_data = raw_data.collect()
final_data = final_data.sort('movie_title')
# cargar df en csv
final_data = final_data.write_csv(r'data/rotten_tomatoes_movies_clean.csv')

# Imprimir la columna de producción
print("Compañías de producción final:")
print(final_data.select('production_company').head(10))