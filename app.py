
import streamlit as st

import pandas as pd

import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk import UnigramTagger, BigramTagger
from nltk.corpus import stopwords
from nltk.corpus import cess_esp 

from unidecode import unidecode 


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


    if 'texto' not in st.session_state:
        st.session_state.texto = ''
    if 'reinicio_titulo' not in st.session_state:
        st.session_state.reinicio_titulo = False
    if 'reinicio_archivo_re' not in st.session_state:
        st.session_state.reinicio_archivo_re = False
    if 'reinicio_archivo_calculo' not in st.session_state:
        st.session_state.reinicio_archivo_calculo = False


    st.write("Ingresa los datos para predecir el tiempo según el modelo entrenado.")
        
    archivo_calculo = st.file_uploader("Carga de archivo CSV", type=["csv"])

    if st.button('Estimar Tiempo de Solución', key="button_calc"):

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





if __name__ == '__main__':
    main()
