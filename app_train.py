
import streamlit as st

import pandas as pd

import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk import UnigramTagger, BigramTagger
from nltk.corpus import stopwords
from nltk.corpus import cess_esp 

from unidecode import unidecode 

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import joblib
#paquetes de espanol 
#nltk.download('stopwords') 
#nltk.download('cess_esp') 
#nltk.download('punkt') 
#nltk.download('punkt_tab')


nlp = spacy.load("es_core_news_sm") 


#stopwords espanol 
stopwords_espanol = stopwords.words('spanish') 
#marcacion verbos... 
oraciones = cess_esp.tagged_sents() 
unigram_tagger = UnigramTagger(oraciones) 
bigram_tagger = BigramTagger(oraciones, backoff=unigram_tagger) 

#preprocesamiento lenguaje 


def aumentar_peso_verbos(palabras, pos_tags, factor=1):
    nuevas_palabras = []
    for palabra, etiqueta in zip(palabras, pos_tags):
            if etiqueta and etiqueta.startswith('v'):
                nuevas_palabras.extend([palabra] * factor)
            else:
                nuevas_palabras.append(palabra)
    return nuevas_palabras

def preprocesar_texto(texto): 

    # Tokenizar y convertir a minúsculas
    palabras = word_tokenize(texto.lower(), language='spanish')
    
    # Etiquetado gramatical
    pos_tags = [tag for _, tag in bigram_tagger.tag(palabras)]
    
    # Aumentar peso de los verbos
    nuevas_palabras = aumentar_peso_verbos(palabras, pos_tags, factor=3)
    
    # Lematización con spaCy
    doc = nlp(" ".join(nuevas_palabras))
    
    # Filtrar y eliminar stopwords
    palabras_filtradas = [
        unidecode(token.lemma_.lower())
        for token in doc
        if token.text.lower() not in stopwords_espanol and token.is_alpha
    ]
    
    return " ".join(palabras_filtradas)




def main():

    st.set_page_config(layout='wide')
    st.title("Estimación de tiempos [GIA]")
    
    if 'texto' not in st.session_state:
        st.session_state.texto = ''
    if 'reinicio_titulo' not in st.session_state:
        st.session_state.reinicio_titulo = False
    if 'reinicio_archivo_re' not in st.session_state:
        st.session_state.reinicio_archivo_re = False
    if 'reinicio_archivo_calculo' not in st.session_state:
        st.session_state.reinicio_archivo_calculo = False


    st.write("Ingresa los datos para reentrenar el modelo.")
    archivo_re = st.file_uploader("Carga de archivo CSV", type=["csv"])
    if st.button('Reentrenar'):
        if archivo_re is not None:

            df_org = pd.read_csv(archivo_re, encoding='utf-8', delimiter=';') 
            df_filt = df_org[['TITULO','IMPACTO','TIEMPO_SOLUCION','GRUPO_ASIGNACION']]
            df_filt = df_filt.dropna(subset=['TIEMPO_SOLUCION']) 
            data_csv = pd.read_csv('train_data.csv', encoding='utf-8', delimiter=';', header=0)
            df_full = pd.concat([df_filt, data_csv]).drop_duplicates()        
            df_full.to_csv('train_data.csv', encoding='utf-8', index=False, sep=';')



            titulos = df_full['TITULO']

            titulos_prepocesados = [preprocesar_texto(tit) for tit in titulos]

            vectorizer = TfidfVectorizer()

            X = vectorizer.fit_transform(titulos_prepocesados)

            df_titulos = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out(),index=df_full.index)

            scaler = StandardScaler()
            df_normalizado = pd.DataFrame(scaler.fit_transform(df_titulos), columns=df_titulos.columns, index=df_titulos.index)

            joblib.dump(vectorizer, 'vectorizer.pkl')
            joblib.dump(scaler, 'scaler.pkl')

            df_B1 = pd.concat([df_full[['TIEMPO_SOLUCION']], df_normalizado], axis=1)

            #print(df_B1)

            #print(preprocesar_texto("Los niños estaban jugando en el parque y viendo qué sucedía a su alrededor."))

            # Variables independientes (X) y dependientes (y)
            X = df_B1.drop(columns=['TIEMPO_SOLUCION'])

            label_encoder = LabelEncoder()
            df_B1['TIEMPO_SOLUCION'] = label_encoder.fit_transform(df_B1['TIEMPO_SOLUCION'])
            joblib.dump(label_encoder, 'label_encoder.pkl')

            y = df_B1['TIEMPO_SOLUCION']

            print(y)
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17)

            # Crear el modelo de Random Forest
            rf_model = RandomForestClassifier(
                n_estimators=100,  # Número de árboles
                max_depth=None,    # Profundidad máxima de los árboles (None para sin límite)
                random_state=17    # Reproducibilidad
            )

            # Entrenar el modelo
            rf_model.fit(X_train, y_train)

            # Realizar predicciones
            y_pred = rf_model.predict(X_test)

            # Evaluar el modelo
            f1_macro = f1_score(y_test, y_pred, average='macro')  # Promedio ponderado
            f1_weighted = f1_score(y_test, y_pred, average='weighted')  # Considerando clases desbalanceadas

            st.success(f"F1 Macro (Tiempo): {f1_macro:.2f}")

            st.success(f"F1 Weighted (Tiempo): {f1_weighted:.2f}")
            st.success("\nClassification Report (Tiempo):")
            st.success(classification_report(y_test, y_pred))


            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(3, 3))  # Crear una figura para la matriz de confusión
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix")

            # Mostrar figura en Streamlit
            st.pyplot(fig)


            joblib.dump(rf_model, 'rf_model.pkl', compress=2)
                 
#######################################################################################################################################################
  
            #df_full = df_filt

            df_B2 = pd.concat([df_full[['GRUPO_ASIGNACION']], df_normalizado], axis=1)


            #print(df_B1)

            #print(preprocesar_texto("Los niños estaban jugando en el parque y viendo qué sucedía a su alrededor."))

            # Variables independientes (X) y dependientes (y)
            X2 = df_B2.drop(columns=['GRUPO_ASIGNACION'])

            label_encoder_GA = LabelEncoder()
            df_B2['GRUPO_ASIGNACION'] = label_encoder_GA.fit_transform(df_B2['GRUPO_ASIGNACION'])
            joblib.dump(label_encoder_GA, 'label_encoder_GA.pkl')

            y2 = df_B2['GRUPO_ASIGNACION']

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.1, random_state=27)

            # Crear el modelo de Random Forest
            rf_model_GA = RandomForestClassifier(
                n_estimators=100,  # Número de árboles
                max_depth=None,    # Profundidad máxima de los árboles (None para sin límite)
                random_state=17    # Reproducibilidad
            )

            # Entrenar el modelo
            rf_model_GA.fit(X2_train, y2_train)

            # Realizar predicciones
            y2_pred = rf_model_GA.predict(X2_test)

            # Evaluar el modelo
            f1_macro2 = f1_score(y2_test, y2_pred, average='macro')  # Promedio ponderado
            f1_weighted2 = f1_score(y2_test, y2_pred, average='weighted')  # Considerando clases desbalanceadas

            st.success(f"F1 Macro (Grupo): {f1_macro2:.2f}")

            st.success(f"F1 Weighted (Grupo): {f1_weighted2:.2f}")
            st.success("\nClassification Report (Grupo):")
            st.success(classification_report(y2_test, y2_pred))


            # Matriz de confusión
            cm = confusion_matrix(y2_test, y2_pred)
            fig, ax = plt.subplots(figsize=(3, 3))  # Crear una figura para la matriz de confusión
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y2))
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix")

            # Mostrar figura en Streamlit
            st.pyplot(fig)

            joblib.dump(rf_model_GA, 'rf_model_GA.pkl', compress=2)

#######################################################################################################################################################

            #df_full = df_filt

            df_B3 = pd.concat([df_full[['IMPACTO']], df_normalizado], axis=1)


            #print(df_B1)

            #print(preprocesar_texto("Los niños estaban jugando en el parque y viendo qué sucedía a su alrededor."))

            # Variables independientes (X) y dependientes (y)
            X3 = df_B3.drop(columns=['IMPACTO'])

            label_encoder_IM = LabelEncoder()
            df_B3['IMPACTO'] = label_encoder_IM.fit_transform(df_B3['IMPACTO'])
            joblib.dump(label_encoder_IM, 'label_encoder_IM.pkl')

            y3 = df_B3['IMPACTO']

            print(y)
            # Dividir los datos en conjuntos de entrenamiento y prueba
            X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.1, random_state=27)

            # Crear el modelo de Random Forest
            rf_model_IM = RandomForestClassifier(
                n_estimators=100,  # Número de árboles
                max_depth=None,    # Profundidad máxima de los árboles (None para sin límite)
                random_state=17    # Reproducibilidad
            )

            # Entrenar el modelo
            rf_model_IM.fit(X3_train, y3_train)

            # Realizar predicciones
            y3_pred = rf_model_IM.predict(X3_test)

            # Evaluar el modelo
            f1_macro3 = f1_score(y3_test, y3_pred, average='macro')  # Promedio ponderado
            f1_weighted3 = f1_score(y3_test, y3_pred, average='weighted')  # Considerando clases desbalanceadas

            st.success(f"F1 Macro (Impacto): {f1_macro3:.2f}")

            st.success(f"F1 Weighted (Impacto): {f1_weighted3:.2f}")
            st.success("\nClassification Report (Impacto):")
            st.success(classification_report(y3_test, y3_pred))


            # Matriz de confusión
            cm = confusion_matrix(y3_test, y3_pred)
            fig, ax = plt.subplots(figsize=(3, 3))  # Crear una figura para la matriz de confusión
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y3))
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
            ax.set_title("Confusion Matrix")

            # Mostrar figura en Streamlit
            st.pyplot(fig)

            joblib.dump(rf_model_IM, 'rf_model_IM.pkl', compress=2)

        else:
            st.warning("Archivo no cargado")

        st.session_state.reinicio_archivo_re = True

    if st.session_state.reinicio_archivo_re:
        if st.button("Reiniciar"):
            st.session_state.reinicio_archivo_re = False
            st.session_state.archivo_re = ''
            archivo_re = None  
            st.rerun()


if __name__ == '__main__':
    main()
