REM missing data imputation
REM "C:\Program Files\R\R-3.5.1\bin\Rscript.exe" amelia_missing_data.R

REM remove all columns with more than half missing data
"C:\Users\z003vnrt\AppData\Local\Continuum\anaconda3\Scripts\ipython3.exe" ugb-data-prep.py

REM KNN imputation
"C:\Users\z003vnrt\AppData\Local\Continuum\anaconda3\Scripts\ipython3.exe" knnimpute_lda_visualization.py

REM prepare data for RNN
"C:\Users\z003vnrt\AppData\Local\Continuum\anaconda3\Scripts\ipython3.exe" prepare_data_for_RNN.py

REM question clustering
"C:\Users\z003vnrt\AppData\Local\Continuum\anaconda3\Scripts\ipython3.exe" question_clustering.py