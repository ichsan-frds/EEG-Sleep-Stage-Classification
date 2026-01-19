import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
import yasa
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import tempfile
import os

st.set_page_config(page_title="Sleep Stage App", layout="wide")

INT_TO_LABEL = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
YASA_TO_INT = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}

def plot_hypnogram(y_pred, title):
    """Fungsi helper untuk plot hypnogram"""
    times = np.arange(len(y_pred)) * 30 / 60 

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(times, y_pred, where='post', color='navy')
    
    ax.set_yticks(list(INT_TO_LABEL.keys()))
    ax.set_yticklabels(list(INT_TO_LABEL.values()))
    ax.invert_yaxis() 

    ax.set_xlabel("Waktu (Menit)")
    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def preprocess_for_eegnet(raw):
    """
    FIXED PIPELINE:
    Menghapus transpose akhir agar shape sesuai input model (Channels, Time).
    Target Shape: (Epochs, 7, 3000, 1)
    """
    raw.filter(0.3, 35.0, fir_design='firwin', verbose=False)

    if raw.info['sfreq'] != 100:
        raw.resample(100)

    data = raw.get_data()
    sfreq = 100

    epoch_size = 30 * sfreq  
    n_epochs = data.shape[1] // epoch_size

    data = data[:, :n_epochs * epoch_size]

    X = data.reshape(data.shape[0], n_epochs, epoch_size)

    X = np.transpose(X, (1, 0, 2))

    n_epochs, n_channels, n_times = X.shape
    X_scaled = np.zeros_like(X)

    for ch in range(n_channels):
        scaler = StandardScaler()
        ch_data = X[:, ch, :].reshape(-1, 1)
        scaler.fit(ch_data)
        X_scaled[:, ch, :] = scaler.transform(
            ch_data).reshape(n_epochs, n_times)

    X_final = X_scaled

    X_final = X_final[..., np.newaxis]

    X_final = X_final.astype(np.float32)

    return X_final

st.title("Sleep Stage Classifier")
st.markdown("Upload file EDF kamu untuk melihat prediksi tahapan tidur.")

st.sidebar.header("Pengaturan")
model_option = st.sidebar.radio(
    "Pilih Model:", ["EEG_net.keras (Custom)", "YASA (Auto)"])
uploaded_file = st.sidebar.file_uploader("Upload file .edf", type=["edf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    st.sidebar.success(f"File '{uploaded_file.name}' terupload!")

    if st.sidebar.button("Mulai Prediksi"):
        try:
            with st.spinner("Membaca file EDF..."):
                raw = mne.io.read_raw_edf(
                    tmp_path, preload=True, verbose=False)
                st.write(
                    f"**Info File:** {len(raw.ch_names)} Channel, Durasi: {raw.times[-1]/60:.1f} Menit")

            if model_option == "EEG_net.keras (Custom)":
                model_path = "EEG_net.keras"

                if not os.path.exists(model_path):
                    st.error(
                        f"File model '{model_path}' tidak ditemukan di folder ini!")
                else:
                    with st.spinner("Preprocessing & Inference EEGNet..."):
                        X_input = preprocess_for_eegnet(raw)

                        model = load_model(model_path)

                        preds_probs = model.predict(X_input)
                        y_pred = preds_probs.argmax(axis=-1)

                        st.subheader("Hasil Prediksi: EEGNet")
                        st.pyplot(plot_hypnogram(
                            y_pred, "Hypnogram (EEGNet Custom)"))

            elif model_option == "YASA (Auto)":
                with st.spinner("Menjalankan YASA (Automated Sleep Staging)..."):
                    eeg_name = None
                    for ch in raw.ch_names:
                        if any(x in ch.lower() for x in ['c3', 'c4', 'fz', 'pz', 'eeg']):
                            eeg_name = ch
                            break

                    if eeg_name is None:
                        eeg_name = raw.ch_names[0]
                        st.warning(
                            f"Channel EEG spesifik tidak ketemu, menggunakan: {eeg_name}")
                    else:
                        st.info(f"YASA menggunakan channel: {eeg_name}")

                    sls = yasa.SleepStaging(raw, eeg_name=eeg_name)
                    y_pred_str = sls.predict()

                    y_pred = np.array([YASA_TO_INT[s] for s in y_pred_str])

                    st.subheader("Hasil Prediksi: YASA")
                    st.pyplot(plot_hypnogram(
                        y_pred, "Hypnogram (YASA Pre-trained)"))

            st.divider()
            st.subheader("Statistik Tahapan Tidur")

            unique, counts = np.unique(y_pred, return_counts=True)
            total_epochs = len(y_pred)

            col1, col2 = st.columns(2)
            with col1:
                df_stats = pd.DataFrame({
                    "Stage": [INT_TO_LABEL[u] for u in unique],
                    "Epochs": counts,
                    "Persentase": [f"{c/total_epochs:.1%} " for c in counts]
                })
                st.table(df_stats)

            with col2:
                fig_pie, ax_pie = plt.subplots()
                ax_pie.pie(counts, labels=[INT_TO_LABEL[u]
                        for u in unique], autopct='%1.1f%%', startangle=90)
                st.pyplot(fig_pie)

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.remove(tmp_path)

else:
    st.info("Upload file di sidebar")
