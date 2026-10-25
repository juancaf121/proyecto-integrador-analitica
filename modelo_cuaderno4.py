"""Transformador del Cuaderno 4: replica la arquitectura PCA + K-Means del grupo.
Código del transformador extraído de `04_Modelos_Supervisados_Regresion_y_Clasificacion.ipynb`.
No se usan PC1, PC2 ni cluster exportados como predictores supervisados.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
RANDOM_STATE = 42

class ExtractorPCAKMeans(BaseEstimator, TransformerMixin):
    """
    Replica exactamente la función `construir_matriz` + `PCA` + `KMeans(k=3)` del Cuaderno 3,
    pero encapsulada en un transformador de scikit-learn ajustado SOLO sobre X_train
    para evitar cualquier fuga de información (Data Leakage):
      - Estandariza 'edad' (y opcionalmente variables cíclicas temporales) con StandardScaler en X_train.
      - Codifica binarias en 0/1 sin estandarizar: sexo_masculino, referido_otra_ips
        (y politraumatismo cuando es predictor en Regresión).
      - Agrupa categorías con frecuencia < 1% en 'otros' (aprendidas en X_train) y genera
        indicadores One-Hot 0/1 sin escalar para las categóricas de PREF_C3.
      - Ajusta PCA en X_train (reteniendo >=80% de varianza si var_threshold=0.80, o el 100% si var_threshold=None).
      - Ajusta K-Means (k=3, n_init=20) sobre los componentes principales al 80% de X_train
        y agrega las variables indicadoras del Clúster K-Means (KMeans_C1, KMeans_C2).
      - Permite dos modos de salida:
          * mode='pca_clusters': devuelve [PC1, PC2, ..., PC_m, KMeans_C1, KMeans_C2]
          * mode='vars_clusters': devuelve [Variables_Matriz_C3 (drop_first=True), KMeans_C1, KMeans_C2]
    """
    def __init__(self, pref_dict=None, include_politrauma=True, include_temporales=False,
                 var_threshold=0.80, n_clusters=3, umbral_rara=0.01, mode='pca_clusters'):
        self.pref_dict = pref_dict
        self.include_politrauma = include_politrauma
        self.include_temporales = include_temporales
        self.var_threshold = var_threshold
        self.n_clusters = n_clusters
        self.umbral_rara = umbral_rara
        self.mode = mode

    def _get_pref_dict(self):
        if self.pref_dict is not None:
            return self.pref_dict
        return {
            'tipo_de_vehiculo': 'veh',
            'relacion_usuario_accidente': 'rol',
            'ips': 'ips',
            'tipo_administradora': 'adm',
            'patologia_principal': 'pat',
            'region_principal': 'reg'
        }

    def _construir_matriz_interna(self, X, is_train=True, drop_first_dummies=False):
        pref = self._get_pref_dict()
        cont_cols = ['edad'] + (['mes_sin', 'mes_cos', 'dia_semana_sin', 'dia_semana_cos'] if self.include_temporales else [])

        if is_train:
            self.scaler_ = StandardScaler()
            Xc_arr = self.scaler_.fit_transform(X[cont_cols].astype(float))
        else:
            Xc_arr = self.scaler_.transform(X[cont_cols].astype(float))
        Xc = pd.DataFrame(Xc_arr, columns=cont_cols, index=X.index)

        # Binarias en 0/1 sin escalar (idéntico al Cuaderno 3)
        dict_bin = {
            'sexo_masculino': (X['sexo'].astype(str).str.lower().str.strip() == 'masculino').astype(float),
            'referido_otra_ips': (X['referido_a_otra_ips'].astype(str).str.lower().str.strip() == 'si').astype(float)
        }
        if self.include_politrauma and 'politraumatismo' in X.columns:
            dict_bin['politraumatismo'] = (X['politraumatismo'].astype(str).str.lower().str.strip() == 'si').astype(float)
        Xb = pd.DataFrame(dict_bin, index=X.index)

        # Categóricas con agrupación de raras (<1% -> 'otros') aprendida solo en X_train
        if is_train:
            self.raras_por_col_ = {}
            self.otros_vacio_ = {}
            for c in pref.keys():
                s = X[c].astype(str)
                frec = s.value_counts(normalize=True)
                raras = frec[frec < self.umbral_rara].index.tolist()
                self.raras_por_col_[c] = raras
                s2 = s.where(~s.isin(raras), 'otros')
                self.otros_vacio_[c] = bool((s2 == 'otros').mean() < self.umbral_rara)

        lista_dummies = []
        for c, p in pref.items():
            s = X[c].astype(str)
            raras = self.raras_por_col_[c]
            s2 = s.where(~s.isin(raras), 'otros')
            if self.otros_vacio_[c]:
                s2 = s2.where(s2 != 'otros', np.nan)
            d = pd.get_dummies(s2, prefix=p, drop_first=drop_first_dummies, dtype=float)
            lista_dummies.append(d)

        Xk = pd.concat(lista_dummies, axis=1)

        if is_train:
            if drop_first_dummies:
                self.dummy_cols_drop_ = Xk.columns.tolist()
            else:
                self.dummy_cols_full_ = Xk.columns.tolist()
        else:
            target_cols = self.dummy_cols_drop_ if drop_first_dummies else self.dummy_cols_full_
            Xk = Xk.reindex(columns=target_cols, fill_value=0.0)

        return pd.concat([Xc, Xb, Xk], axis=1)

    def fit(self, X, y=None):
        # 1. Matriz completa (sin drop_first, igual a X_final del Cuaderno 3) para PCA y K-Means
        X_full = self._construir_matriz_interna(X, is_train=True, drop_first_dummies=False)
        self.full_feature_names_ = X_full.columns.tolist()

        # 2. Ajuste de PCA completo en X_train y determinación del corte al umbral de varianza
        self.pca_full_ = PCA(random_state=RANDOM_STATE).fit(X_full.values)
        va = np.cumsum(self.pca_full_.explained_variance_ratio_)
        self.n_pcs_80_ = int(np.argmax(va >= 0.80) + 1)

        if self.var_threshold is not None:
            self.n_pcs_ = int(np.argmax(va >= self.var_threshold) + 1)
        else:
            # Retiene todos los componentes con autovalor positivo (> 1e-8) para evitar colinealidad exacta de dummies
            self.n_pcs_ = int(np.sum(self.pca_full_.explained_variance_ > 1e-8))

        self.var_explicada_ = float(va[self.n_pcs_ - 1])
        Z_train_full = self.pca_full_.transform(X_full.values)

        # 3. Ajuste de K-Means (k=3, n_init=20) sobre los primeros n_pcs_80_ componentes (idéntico al Cuaderno 3)
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init=20, random_state=RANDOM_STATE)
        self.kmeans_.fit(Z_train_full[:, :self.n_pcs_80_])

        # 4. Si mode == 'vars_clusters', preparamos las columnas con drop_first=True para evitar trampa de variables ficticias en OLS
        if self.mode == 'vars_clusters':
            X_drop = self._construir_matriz_interna(X, is_train=True, drop_first_dummies=True)
            self.vars_feature_names_ = X_drop.columns.tolist()
            self.feature_names_out_ = self.vars_feature_names_ + [f'KMeans_C{c}' for c in range(1, self.n_clusters)]
        else:
            self.feature_names_out_ = [f'PC{i+1}' for i in range(self.n_pcs_)] + [f'KMeans_C{c}' for c in range(1, self.n_clusters)]

        return self

    def transform(self, X):
        X_full = self._construir_matriz_interna(X, is_train=False, drop_first_dummies=False)
        Z_full = self.pca_full_.transform(X_full.values)
        clusters = self.kmeans_.predict(Z_full[:, :self.n_pcs_80_])

        cl_dummies = np.zeros((len(X), self.n_clusters - 1), dtype=float)
        for c in range(1, self.n_clusters):
            cl_dummies[:, c - 1] = (clusters == c).astype(float)

        if self.mode == 'pca_clusters':
            return np.hstack([Z_full[:, :self.n_pcs_], cl_dummies])
        else:
            X_drop = self._construir_matriz_interna(X, is_train=False, drop_first_dummies=True)
            return np.hstack([X_drop.values, cl_dummies])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
