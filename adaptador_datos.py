"""Adaptación explícita de los tres esquemas de entrada para la aplicación.

No reconstruye variables clínicas o administrativas ausentes ni traslada
clústeres de una muestra a otra. Los CSV de origen se mantienen intactos.
"""
from __future__ import annotations
import re
import unicodedata
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PREFIJOS = {
    'tipo_de_vehiculo': 'veh',
    'relacion_usuario_accidente': 'rol',
    'ips': 'ips',
    'tipo_administradora': 'adm',
    'patologia_principal': 'pat',
    'region_principal': 'reg',
}
MESES = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto',
         'septiembre','octubre','noviembre','diciembre']
DIAS = ['lunes','martes','miercoles','jueves','viernes','sabado','domingo']


def _slug(t):
    t = ''.join(ch for ch in unicodedata.normalize('NFKD', str(t)) if not unicodedata.combining(ch))
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9]+', '_', t.casefold())).strip('_')


# Corrige variantes de los encabezados originales; no inventa predictores.
ALIAS = {
    'numero': 'numero',
    'relacion_usuario_accidente': 'relacion_usuario_accidente',
    'fecha_de_atencion_medica': 'fecha_de_atencion_medica',
    'region_anatomica_mas_afectada': 'region_anatomica_mas_afectada',
    'ano': 'ano', 'dia_semana': 'dia_semana',
    'codigo_administradora':'codigo_administradora',
    'tipo_administradora':'tipo_administradora',
    'coigo_rips': 'coigo_rips',
    'descripcion_rips': 'descripcion_rips',
}


def normalizar_datos(tabla: pd.DataFrame):
    """Devuelve (datos normalizados, observaciones, esquema).

    La normalización resuelve encabezados y codificaciones, no crea valores
    para patologia_principal, region_principal, rol ni administrador inexistentes.
    """
    if tabla is None or tabla.empty:
        raise ValueError('El CSV está vacío.')
    df = tabla.copy()
    originales = list(df.columns)
    nuevo = [ALIAS.get(_slug(c), _slug(c)) for c in originales]
    if len(nuevo) != len(set(nuevo)):
        raise ValueError('Existen encabezados duplicados luego de normalizar nombres.')
    df.columns = nuevo
    hallazgos = []
    if 'patologia_principal' in df.columns and 'region_principal' in df.columns:
        esquema = 'Modelado preparado'
    elif 'descripcion_rips' in df.columns and 'relacion_usuario_accidente' in df.columns:
        esquema = 'Original cruda'
    elif 'descripcion_rips' in df.columns:
        esquema = 'Limpia intermedia'
    else:
        esquema = 'CSV compatible parcial'

    objetos = list(df.select_dtypes(include=['object', 'string']).columns)
    for col in objetos:
        df[col] = df[col].astype('string').str.strip().str.lower()
        df[col] = df[col].replace({'': pd.NA, 'nan': pd.NA, 'none': pd.NA})

    if 'costos' in df.columns:
        costos = df['costos']
        if not pd.api.types.is_numeric_dtype(costos):
            costos = costos.astype('string').str.replace('$', '', regex=False).str.replace(' ', '', regex=False)
            # Separadores de miles cuando el origen usa números tipo 1.234.567.
            costos = costos.str.replace(r'(?<=\d)\.(?=\d{3}(?:\D|$))', '', regex=True)
            costos = costos.str.replace(',', '.', regex=False)
        df['costos'] = pd.to_numeric(costos, errors='coerce')
        precios_validos = (df['costos'] > 0).sum()
        if precios_validos < len(df):
            hallazgos.append(f'{len(df)-precios_validos:,} registros tienen costos nulos/no válidos o menores o iguales a cero; se excluyen solo de los análisis que requieren log₁₀(costos).')
        with np.errstate(divide='ignore', invalid='ignore'):
            df['log_costos'] = np.log10(df['costos'].where(df['costos'] > 0))

    if 'edad' in df.columns:
        df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    if 'ano' in df.columns:
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce').astype('Int64')
    if 'politraumatismo' in df.columns:
        serie = df['politraumatismo'].astype('string').str.normalize('NFKD').str.replace(r'[\u0300-\u036f]', '', regex=True)
        df['politraumatismo'] = serie.replace({'1': 'si', 'true': 'si', 'yes': 'si', '0': 'no', 'false': 'no'})

    if 'mes' in df.columns:
        df['mes'] = df['mes'].astype('string').str.replace(r'^\s*\d+\s*\.\s*', '', regex=True)
    if 'dia_semana' in df.columns:
        df['dia_semana'] = df['dia_semana'].astype('string').str.replace(r'^\s*\d+\s*\.\s*', '', regex=True)
        df['dia_semana'] = df['dia_semana'].str.replace('á', 'a').str.replace('é', 'e')

    if 'mes_numeric' not in df.columns:
        if 'mes' in df.columns:
            df['mes_numeric'] = df['mes'].map({v:i+1 for i,v in enumerate(MESES)}).astype(float)
        elif 'fecha_de_accidente' in df.columns:
            df['mes_numeric'] = pd.to_datetime(df['fecha_de_accidente'], errors='coerce').dt.month.astype(float)
    if 'dia_semana_numeric' not in df.columns:
        if 'dia_semana' in df.columns:
            df['dia_semana_numeric'] = df['dia_semana'].map({v:i+1 for i,v in enumerate(DIAS)}).astype(float)
        elif 'fecha_de_accidente' in df.columns:
            df['dia_semana_numeric'] = (pd.to_datetime(df['fecha_de_accidente'], errors='coerce').dt.dayofweek + 1).astype(float)
    if 'mes_numeric' in df.columns:
        mes = pd.to_numeric(df['mes_numeric'], errors='coerce')
        if 'mes_sin' not in df: df['mes_sin'] = np.sin(2*np.pi*mes/12)
        if 'mes_cos' not in df: df['mes_cos'] = np.cos(2*np.pi*mes/12)
    if 'dia_semana_numeric' in df.columns:
        dia = pd.to_numeric(df['dia_semana_numeric'], errors='coerce')
        if 'dia_semana_sin' not in df: df['dia_semana_sin'] = np.sin(2*np.pi*dia/7)
        if 'dia_semana_cos' not in df: df['dia_semana_cos'] = np.cos(2*np.pi*dia/7)

    cat_modelo = ['sexo', 'referido_a_otra_ips', *PREFIJOS]
    for col in cat_modelo:
        if col in df.columns:
            df[col] = df[col].astype('string').fillna('sin_informacion').astype(str)

    ausentes = [c for c in PREFIJOS if c not in df]
    if ausentes:
        hallazgos.append('Predictores ausentes que NO se reconstruyen: ' + ', '.join(ausentes) + '. Los modelos se reajustan con las variables disponibles.')
    if esquema != 'Modelado preparado':
        hallazgos.append('Los resultados de esta base no son directamente comparables con los del Entregable 4: cambian la muestra, el preprocesamiento y/o los predictores.')
    return df, hallazgos, esquema


def predictores_disponibles(df):
    base = {'edad', 'sexo', 'referido_a_otra_ips', 'politraumatismo'}
    faltan = sorted(base.difference(df.columns))
    if faltan: raise ValueError('Columnas esenciales faltantes: ' + ', '.join(faltan))
    pref = {c: p for c, p in PREFIJOS.items() if c in df.columns}
    if not pref: raise ValueError('No hay variables categóricas predictoras útiles.')
    return pref


def _matriz_descriptiva(df):
    """Matriz PCA en el mismo espíritu del Entregable 3, sin fuga hacia modelos supervisados."""
    d = df.copy()
    d['edad'] = pd.to_numeric(d['edad'], errors='coerce')
    d['costos'] = pd.to_numeric(d['costos'], errors='coerce')
    d = d.loc[d['costos'].gt(0) & d['edad'].notna() & d['politraumatismo'].isin(['si', 'no'])].copy()
    if len(d) < 30: raise ValueError('Hay menos de 30 filas válidas para PCA y K-means.')
    d['log_costos'] = np.log10(d['costos'])
    c = pd.DataFrame(StandardScaler().fit_transform(d[['edad','log_costos']]),
                     columns=['edad','log_costos'], index=d.index)
    b = pd.DataFrame({
        'sexo_masculino': d['sexo'].eq('masculino').astype(float),
        'referido_otra_ips': d['referido_a_otra_ips'].eq('si').astype(float),
        'politraumatismo': d['politraumatismo'].eq('si').astype(float),
    }, index=d.index)
    cats = []
    for col, pref in PREFIJOS.items():
        if col not in d.columns: continue
        s = d[col].fillna('sin_informacion').astype(str)
        raras = s.value_counts(normalize=True).loc[lambda x: x < 0.01].index
        s = s.where(~s.isin(raras), 'otros')
        cats.append(pd.get_dummies(s, prefix=pref, dtype=float))
    bloques = [c,b]+cats
    X = pd.concat(bloques, axis=1).astype(float)
    return d, X


def calcular_segmentos(df):
    """PCA/K-means independiente de la base oficial; agrupamiento descriptivo."""
    d,X = _matriz_descriptiva(df)
    pca = PCA(random_state=42, svd_solver='full').fit(X.values)
    var = np.cumsum(pca.explained_variance_ratio_)
    m = int(np.searchsorted(var, .80) + 1)
    z = pca.transform(X.values)
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(z[:, :m])
    seg = d.copy()
    seg['PC1'], seg['PC2'], seg['cluster'] = z[:,0], z[:,1], km.labels_
    cargas = pd.DataFrame(pca.components_[:min(3,len(pca.components_))].T,
                          index=X.columns, columns=[f'PC{i+1}' for i in range(min(3,len(pca.components_)))])
    return {'segmentos':seg, 'n_pca_80':m, 'varianza_retenida':float(var[m-1]),
            'varianza_acumulada':var.tolist(), 'cargas':cargas,
            'pca_80':z[:,:m], 'columnas_entrada':list(X.columns)}


def diagnostico_k(data):
    """Diagnóstico exploratorio muestral, calculado sobre los datos elegidos."""
    z = data['pca_80']
    if len(z) > 3500:
        ix = np.random.default_rng(42).choice(len(z), 3500, replace=False)
        z = z[ix]
    diagnosticos=[]
    for k in range(2, min(7,len(z))):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(z)
        ss = silhouette_score(z, km.labels_, sample_size=min(1200,len(z)), random_state=42)
        diagnosticos.append({'k':k, 'inercia':float(km.inertia_), 'silhouette':float(ss)})
    return pd.DataFrame(diagnosticos)
