
import streamlit as st

import pandas as pd

import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk import UnigramTagger, BigramTagger
from nltk.corpus import stopwords
from nltk.corpus import cess_esp 

from unidecode import unidecode 


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

import joblib
#paquetes de espanol 
nltk.download('stopwords') 
nltk.download('cess_esp') 
nltk.download('punkt') 
nltk.download('punkt_tab')


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
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    vectorizer = joblib.load('vectorizer.pkl')

    st.set_page_config(layout='wide')
    st.title("Estimación de tiempos [GIA]")

    st.sidebar.markdown(
        """
        <style>
        .sidebar .sidebar-content{
            font-size: 52px
        }
        </style>
        """,unsafe_allow_html=True
    )

    menu = st.sidebar.radio("Modo de uso", ["Calculo individual","Calculo por archivo","Reentrenamiento de modelo de prediccion"])
    
    if 'texto' not in st.session_state:
        st.session_state.texto = ''
    if 'reinicio_titulo' not in st.session_state:
        st.session_state.reinicio_titulo = False
    if 'reinicio_archivo_re' not in st.session_state:
        st.session_state.reinicio_archivo_re = False
    if 'reinicio_archivo_calculo' not in st.session_state:
        st.session_state.reinicio_archivo_calculo = False

    if menu == "Calculo individual":
        st.write("Ingresa los datos para predecir el tiempo según el modelo entrenado.")
    

        user_input_org = st.text_input('Titulo', st.session_state.texto)
        st.session_state.texto = user_input_org

        if st.button('Estimar Tiempo Máximo de Solución'):
            if user_input_org:

                user_input = preprocesar_texto(user_input_org)

                user_vector = vectorizer.transform([user_input])
                df_titulos = pd.DataFrame(user_vector.toarray(), columns=vectorizer.get_feature_names_out())

                #df_servicio = pd.DataFrame({"SERVICIO":[servicio]})
                #datos_kluster = pd.concat([df_servicio, df_titulos],axis = 1)

                scaled_vector = scaler.transform(df_titulos.select_dtypes(include=[float,int]))

                cluster = model.predict(scaled_vector)[0]

                st.success(f"El tiempo estimado de la tarea es de : {cluster} minutos")
            else:
                st.warning("Titulo no definido")

            st.session_state.reinicio_titulo = True

        if st.session_state.reinicio_titulo:
            if st.button("Reiniciar"):
                st.session_state.reinicio_titulo = False
                st.session_state.texto = ''
                user_input_org = ''   
                st.rerun()



    elif menu == "Calculo por archivo":
        st.write("Ingresa los datos para predecir el tiempo según el modelo entrenado.")
        
        archivo_calculo = st.file_uploader("Carga de archivo CSV", type=["csv"])

        if st.button('Estimar Tiempo de Solución'):

            if archivo_calculo is not None:
                st.session_state.archivo_calculo = archivo_calculo

                df = pd.read_csv(archivo_calculo, encoding='utf-8', delimiter=',')     
                datos_rf = df
                titulos = datos_rf['Descripción'] 
                documentos_procesados = [preprocesar_texto(doc) for doc in titulos]
                X = vectorizer.transform(documentos_procesados)
                df_titulos = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
                
                df_normalizado = scaler.transform(df_titulos.select_dtypes(include=[float,int]))
                tiempo_calc = model.predict(df_normalizado)
                datos_rf['Tiempo_estimado']=tiempo_calc
                #datos_filtrados['Tiempo_estimado']=datos_filtrados.groupby('K')['TIEMPO_SOLUCION'].transform(lambda x: int(x.mean()) if x.notna().any() else 450)
                #datos_filtrados.to_csv('resultado.csv',index=False,sep='|')
                csv = datos_rf.to_csv(index=False,sep=';')


                st.dataframe(datos_rf, use_container_width=True)

                st.download_button(
                    label="Descargar resultados",
                    data=csv,
                    file_name="datos.csv",
                    mime="texto/csv"
                )

            else:
                st.warning("Archivo no cargado")


            st.session_state.reinicio_archivo_calculo = True

        if st.session_state.reinicio_archivo_calculo:
            if st.button("Reiniciar"):
                st.session_state.reinicio_archivo_calculo = False
                st.session_state.archivo_calculo = ''
                archivo_calculo = None  
                st.rerun()






    elif menu == "Reentrenamiento de modelo de prediccion":
        st.write("Ingresa los datos para reentrenar el modelo.")
        archivo_re = st.file_uploader("Carga de archivo CSV", type=["csv"])
        if st.button('Reentrenar'):
            if archivo_re is not None:
                df_org = pd.read_csv(archivo_re, encoding='utf-8', delimiter=';') 
                df_filt = df_org[['TITULO','IMPACTO','TIEMPO_SOLUCION']]
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

                df_B1 = pd.concat([df_full[['TIEMPO_SOLUCION']], df_normalizado], axis=1)


                joblib.dump(vectorizer, 'vectorizer.pkl')
                joblib.dump(scaler, 'scaler.pkl')

                #print(df_B1)

                #print(preprocesar_texto("Los niños estaban jugando en el parque y viendo qué sucedía a su alrededor."))

                # Variables independientes (X) y dependientes (y)
                X = df_B1.drop(columns=['TIEMPO_SOLUCION'])
                y = df_B1['TIEMPO_SOLUCION']

                print(y)
                # Dividir los datos en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=17)

                # Crear el modelo de Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=170,  # Número de árboles
                    max_depth=None,    # Profundidad máxima de los árboles (None para sin límite)
                    random_state=17    # Reproducibilidad
                )

                # Entrenar el modelo
                rf_model.fit(X_train, y_train)

                # Realizar predicciones
                y_pred = rf_model.predict(X_test)

                # Evaluar el modelo
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                joblib.dump(rf_model, 'rf_model.pkl')
                
                st.success(f"Mean Squared Error (MSE): {mse:.2f}")

                st.success(f"R-squared (R²): {r2:.2f}")

                st.dataframe(df_full, use_container_width=True)

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