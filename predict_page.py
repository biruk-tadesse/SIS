from turtle import width
import streamlit as st
import pickle
import numpy as np
import os
import librosa
from speakerfeatures import extract_features
from sound import record

def load_model():
    path = "speaker_models"
    gmm_files = [os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith('.gmm')]
    models = [pickle.load(open(file_namename, 'r+b')) for file_namename in gmm_files]
    return models

models = load_model()

def show_predict_page():
    st.set_page_config(page_title="የተናጋሪ መለያ ስርዓት / Speaker Identification System",layout='wide')
    padding = 0
    st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}1rem;
        padding-left: {padding}1rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
    col1, col2 = st.beta_columns([1,2.5])
    with col1:
        st.markdown("***")
        st.image('ALX.webp', width=350)
        url = "https://www.alxafrica.com//"
        st.markdown("[ALX Africa](%s)" % url)
        st.markdown("[ALX አፍሪካ](%s)" % url)
        st.markdown("***")
        st.text("© ALX")
        st.text("2023/2015")

    with col2:
        st.markdown(f'<h1 style="color:#0068c9;font-size:30px;">{"የተናጋሪ መለያ ስርዓት / Speaker Identification System"}</h1>', unsafe_allow_html=True)
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
        st.markdown("- ይህ ሲስተም በድምጽ ፋይል ውስጥ የሚናገር ሰው ማንነትን ለመለየት ይጠቅማል፡፡ / This system is used to identify the person speaking in an audio file.")
        st.markdown("- ሲስተሙ የተቀዳ የድምጽ ፋይልን በመጫን ወይም አዲስ ድምጽ በመቅዳት ይሰራል:: / The system works by loading a recorded sound file or recording a new sound.")
        st.markdown("- ሲስተሙ  ሰው ሰራሽ የማሰብ ችሎታ (AI)ን ይጠቀማል:: ስለዚህ በቅድሚያ በራስዎ ዳታሴት መሰልጠን ያስፈልገዋል:: / The system uses artificial intelligence (AI). So it needs to be trained first.")
        st.markdown(
        """
        <style>
            .css-9ycgxx::after {
                content: "/ የድምጽ ፋይሉን እዚህ ይጫኑ / Upload the audio file here";
            }
        <style>
        """, unsafe_allow_html=True)

        speakers = [speaker.split('.')[0] for speaker in os.listdir("speaker_models")]
        uploaded_audio = st.file_uploader("")
        if uploaded_audio:
            st.audio(uploaded_audio)
            signal, sr = librosa.load(uploaded_audio)
            print(signal.shape, sr)
            X = extract_features(signal, sr)
        if st.button("ቅዳ/Record"):
            with st.spinner(f'{5} ሰከንዶች በመቅዳት ላይ / 5 Seconds Recording ....'):
                record()
            st.success("ቅጂ ተጠናቋል / Recording Completed::")
            st.audio("record.wav")
        if st.button('ተንብይ / Predict'):
            if not uploaded_audio:
                signal, sr = librosa.load("record.wav")
                print(signal.shape, sr)
                X = extract_features(signal, sr)
            log_likelihood = np.zeros(len(models))
            for i in range(len(models)):
                gmm = models[i]  # checking with each model one by one
                scores = np.array(gmm.score(X))
                log_likelihood[i] = scores.sum()
            winner = np.argmax(log_likelihood)
            st.markdown('**የዚህ ድምፅ ባለቤት (The speaker is) ' + speakers[winner] + '**' + ".")
