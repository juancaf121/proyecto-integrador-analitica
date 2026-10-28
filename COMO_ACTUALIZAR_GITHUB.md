# Actualizar el proyecto en GitHub y Streamlit Community Cloud

Esta versión procesa sin alterar sus CSV:

1. `data/df_modelado.csv` (28.276 filas; versión de los modelos del grupo).
2. `data/BBDDlimpia.csv` (18.058 filas; versión limpia intermedia).
3. `data/BBDDcruda_original.csv` (34.306 filas; exportación cruda original).

## Subir actualización al repositorio ya existente

- Desde GitHub, abrir el repositorio del grupo y la rama `main`.
- Usar **Add file > Upload files**.
- Arrastrar **el contenido de la carpeta del proyecto**, no el ZIP ni una carpeta contenedora. Esto deja `app.py` y `adaptador_datos.py` en la raíz, y los tres archivos de ejemplo dentro de `data/`.
- Si GitHub notifica que va a sustituir `app.py`, `README.md` o `.gitignore`, es lo esperado. Evitar borrar otros archivos que el grupo ya tenía en el repositorio.
- Confirmar el cambio (commit) con un mensaje como `Compatibilidad Streamlit con base cruda y limpia`.
- Abrir la app desplegada en Streamlit Cloud: debería actualizarse al nuevo commit. En caso de error, revisar **Manage app > Logs** y reiniciar la app desde su panel si fuese necesario.

## Probar en la app

1. En la barra lateral, seleccionar **Base de datos incluida** y elegir la base **cruda original**.
2. Entrar a **PCA y K-means** y verificar tres clústeres. Repetir con **Base limpia intermedia**.
3. Ejecutar **Regresión de costos** y **Clasificación** en ambas bases. Los resultados se calculan en vivo y deben diferir de los del dataset oficial.
4. También se puede subir por el cargador lateral un archivo CSV con encabezados originales en mayúsculas o encabezados normalizados. Al cargar un archivo manual, este tiene prioridad sobre el selector de bases incluidas.
5. Para volver al experimento del grupo, seleccionar **Modelado oficial (28.276)** y quitar cualquier archivo cargado manualmente desde el widget.

## Alcance estadístico

- Los clústeres de la base cruda y limpia se recalculan con la base activa; **no** reutilizan etiquetas del dataset oficial.
- En el dataset limpio faltan `relacion_usuario_accidente`, `tipo_administradora`, `patologia_principal`, `region_principal`; esas cuatro variables **no se fabrican**.
- En el crudo faltan `patologia_principal` y `region_principal`, por lo que se ajustan modelos sin esos dos predictores.
- La base cruda tiene costos cero/no válidos. Los modelos que necesitan `log₁₀(costos)` usan únicamente los registros con costos positivos; la clasificación puede conservar otros registros si su etiqueta y predictores son válidos.
- Si los datos incluidos fueran reales y no sintéticos, revisar su permiso de uso antes de subirlos a repositorios públicos.
