import ast, sys, time, pathlib
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from modelo_cuaderno4 import ExtractorPCAKMeans
from adaptador_datos import *
base=pathlib.Path(__file__).parent
source=ast.parse((base/'app.py').read_text())
node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='evaluar_modelo')
node.decorator_list=[]
ns=globals().copy();ns['RANDOM_STATE']=42
exec(compile(ast.Module(body=[node],type_ignores=[]),str(base/'app.py'),'exec'),ns)
evaluar_modelo=ns['evaluar_modelo']
for sourcefile in ['df_modelado.csv','BBDDlimpia.csv','BBDDcruda_original.csv']:
 start=time.time(); df=pd.read_csv(base/'data'/sourcefile,low_memory=False); df,notes,kind=normalizar_datos(df)
 print('\nSOURCE',sourcefile,len(df),len(df.columns),kind,'NOTES',notes,'time',round(time.time()-start,2),flush=True)
 assert df['politraumatismo'].isin(['si','no']).sum() > 100
 assert df.costos.gt(0).sum() > 100
 for objetivo, modelo in [('Costos','PCA 80 % + lineal'),('Politraumatismo','Árbol de decisión')]:
  start=time.time(); r=evaluar_modelo(df,objetivo,modelo,5)
  print(' MODEL',objetivo,r['entrenamiento'],r['prueba'],r['n_pc'],r['variables_modelo'],'time',round(time.time()-start,2),flush=True)
 start=time.time(); a=calcular_segmentos(df)
 print(' PCA',len(a['segmentos']),a['n_pca_80'],round(a['varianza_retenida'],3), 'time',round(time.time()-start,2),flush=True)
 start=time.time(); e=diagnostico_k(a)
 print(' SIL',e[['k','silhouette']].round(3).to_dict('records'), 'time',round(time.time()-start,2),flush=True)
