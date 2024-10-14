import polars as pl
from rapidfuzz import process, fuzz
import re
from collections import Counter

# Función para transformar columnas a cadena
def transform_to_string(df, columns):
    """
    Transform type of data in specified columns to string by sorting elements and joining them.
    """
    for col in columns:
        df = df.with_column(
            pl.col(col).apply(lambda x: ', '.join(sorted(x)) if isinstance(x, frozenset) else x).alias(col)
        )
    return df

# Normalización con fuzzy matching
def normalize_category_fuzzy(value, choices, threshold=80):
    """
    Normalizes a given string value by finding the closest match in a list of choices using fuzzy matching.
    """
    if isinstance(value, str):
        best_match = process.extractOne(value, choices)
        if best_match and best_match[1] >= threshold:
            return best_match[0]
    return 'nr'

# Rellenar valores nulos con una fecha ficticia
def fill_date_null(df, dates, date_fict):
    """
    Fill null dates in specified columns of a DataFrame with a fictitious date.
    """
    for col in dates:
        df = df.with_columns(
            pl.col(col).str.strptime(pl.Date, "%Y-%m-%d", strict=False).fill_null(date_fict).alias(col)
        )
    return df

# Preprocesar nombres de compañías con expresiones vectorizadas en lugar de map_elements
def preprocess_company(df, column):
    df = df.with_columns(
        pl.col(column)
        .str.to_lowercase()
        .str.strip_chars()
        .str.replace_all(r'\b(inc|corp|corporation|limited|ltd|plc|pictures|picture|co|entertainment|films|film|studios|corporat)\b', '')
        .str.replace_all(r'[^\w\s]', '')
        .str.replace_all(r'\s+', ' ')
        .str.strip_chars()
        .alias(column)
    )
    return df

# Agrupar compañías similares
def group_similar(company, all_companies, threshold=85):
    """
    Finding the best matches from a list of company names using fuzzy matching.
    """
    matches = process.extract(company, all_companies, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
    if matches:
        match_names = [match[0] for match in matches]
        most_common = Counter(match_names).most_common(1)
        return most_common[0][0] if most_common else company
    return company

original_data = pl.read_csv(r'data\rotten_tomatoes_movies.csv')
raw_data = original_data.clone()

# Definir las categorías y columnas
categorical = ['content_rating', 'tomatometer_status', 'audience_status']
irrelevant = ['rotten_tomatoes_link', 'movie_info', 'critics_consensus', 'tomatometer_count',
              'tomatometer_top_critics_count', 'tomatometer_fresh_critics_count', 'tomatometer_rotten_critics_count', 'audience_count']
relevant = ['movie_title']
objs = ['directors', 'genres', 'authors', 'actors', 'production_company']
numeric = ['runtime']
ratings = ['tomatometer_rating', 'audience_rating']
dates = ['original_release_date', 'streaming_release_date']

# definition of categories
rating   = ['g', 'nc17', 'nr', 'pg', 'pg13', 'r']
t_status = ['rotten', 'fresh', 'certified-fresh', 'nr']
a_status = ['spilled', 'upright', 'nr']

# Eliminar columnas irrelevantes
raw_data = raw_data.drop(irrelevant)

# Eliminar filas con valores faltantes en columnas relevantes
raw_data = raw_data.drop_nulls(relevant)

#Eliminar duplicados
raw_data = raw_data.unique(subset=['movie_title','original_release_date', 'directors']) 

# Rellenar valores nulos en categorías
for col in categorical:
    raw_data = raw_data.with_columns(pl.col(col).fill_null('nr').alias(col).cast(pl.Categorical))
    
# Rellenar valores nulos en objetos y numéricos
raw_data = raw_data.with_columns([pl.col(col).fill_null('nr').alias(col) for col in objs])
raw_data = raw_data.with_columns([pl.col(col).fill_null(0.0).alias(col) for col in numeric])

# Rellenar fechas ficticias
amount_date_nulls = raw_data.select([pl.col(col).is_null().sum().alias(col) for col in dates]).sum().sum()
raw_data = fill_date_null(raw_data, dates, '1900-01-01')

# Normalizar los nombres de las compañías
raw_data = preprocess_company(raw_data, 'production_company')

# Obtener la lista de compañías para usar en la agrupación
companies_list = raw_data['production_company'].to_list()

# Agrupar compañías similares utilizando la lista de compañías
raw_data = raw_data.with_columns(
    pl.col('production_company').map_elements(lambda x: group_similar(x, companies_list, threshold=85), return_dtype=pl.Utf8).alias('production_company')
)

print(amount_date_nulls)
print(raw_data.head())