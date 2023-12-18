from tensorflow.keras.models import load_model
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from clean import downsample_mono, envelope
import numpy as np
import os
from pymongo import MongoClient
import threading
from concurrent.futures import ThreadPoolExecutor
import gc
import tensorflow as tf


def calculate_average_rms_db(audio_segment):
    # Berechnung des Durchschnitts des quadratischen Mittelwerts (Root Mean Square, RMS) für das gesamte Fenster
    rms_value = np.sqrt(np.mean(np.square(audio_segment)))
    # Umrechnung des RMS-Werts in Dezibel und Runden auf 2 Nachkommastellen
    rms_db = round(20 * np.log10(rms_value), 2)
    return rms_db

def convert_numpy_to_python(data):
    if isinstance(data, np.float32) or isinstance(data, np.float64):
        # Runden auf 2 Nachkommastellen für Fließkommazahlen
        return round(float(data), 2)
    elif isinstance(data, np.generic):
        return np.asscalar(data)
    elif isinstance(data, dict):
        return {k: convert_numpy_to_python(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python(v) for v in data]
    else:
        return data


def process_file(audio_file_path, models, collection):
    filename = os.path.basename(audio_file_path)
    
    # Überprüfen, ob die Datei bereits in der MongoDB vorhanden ist
    if collection.find_one({'Filename': filename}):
        print(f"Datei {filename} bereits in der MongoDB vorhanden, überspringe die Verarbeitung.")
        return

    print(f"Verarbeite Datei: {filename}")
    
    # Audioverarbeitung
    rate, wav = downsample_mono(audio_file_path, 32000)
    mask, env = envelope(wav, rate, threshold=20)
    clean_wav = wav[mask]

    # Parameter
    dt = 3.0  # Zeit in Sekunden für die Audio-Probe
    step = int(rate * dt)  # 1 Sekunde Schrittgröße
    window = step  # 3 Sekunden Fenstergröße

    # Initialisiere leere Liste für Klassifikationsergebnisse und Statistiken
    classification_results = []

    stats = []

    for i in range(0, clean_wav.shape[0] - window + 1, step):
        sample = clean_wav[i:i+window]
        sample = sample.astype(np.int32)
        average_rms_db = calculate_average_rms_db(sample)
        sample = sample.reshape(-1, 1)
        X_batch = np.array([sample], dtype=np.float32)

        # Konvertierung des NumPy-Arrays in einen TensorFlow-Tensor
        X_batch_tensor = tf.convert_to_tensor(X_batch, dtype=tf.float32)
        
        results = {}
        for model_name, model in models.items():
            # Verwendung des Tensors für die Vorhersage
            y_pred = model.predict(X_batch_tensor)
            y_mean = np.mean(y_pred, axis=0)
            predicted_label = np.argmax(y_mean)
            confidence_rounded = round(y_mean[predicted_label], 2)
            results[model_name] = {
                'Predicted Class': predicted_label,
                'Confidence': confidence_rounded
            }

        classification_result = {
            'Filename': os.path.basename(audio_file_path),
            'Seconds': i / rate,
            'From_To': f"{i/rate}-{(i+window)/rate}",
            'RMS_dB': average_rms_db,
            'Models': results
        }
        
        classification_results.append(classification_result)
                    # Löschen von Variablen, die nicht mehr benötigt werden
        del sample, X_batch, y_pred, y_mean

        # Aufrufen des Garbage Collectors
        gc.collect()

    # Sammeln der Klassifikationsergebnisse für jedes File
    audio_classification_results = {}

    for result in classification_results:
        filename = result['Filename']
        if filename not in audio_classification_results:
            audio_classification_results[filename] = {
                'Filename': filename,
                'Results': []
            }
        
        # Entfernen des 'Filename' Schlüssels aus dem Ergebnis, bevor es zur Liste hinzugefügt wird
        result_without_filename = {key: value for key, value in result.items() if key != 'Filename'}
        audio_classification_results[filename]['Results'].append(result_without_filename)

    # Konvertieren der Numpy-Datentypen und Schreiben der Daten in MongoDB
    for filename, data in audio_classification_results.items():
        data['Results'] = convert_numpy_to_python(data['Results'])
        collection.insert_one(data)

    print("Klassifikationsergebnisse wurden erfolgreich in die MongoDB geschrieben.")
    del clean_wav, mask, env, classification_results, audio_classification_results
    gc.collect()


def process_audio_directory(mongo_uri, db_name, collection_name, model_paths, root_directory_path):
    # MongoDB Verbindung einrichten
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Modelle laden
    models = {name: load_model(path, custom_objects={
        'STFT': STFT,
        'Magnitude': Magnitude,
        'ApplyFilterbank': ApplyFilterbank,
        'MagnitudeToDecibel': MagnitudeToDecibel
    }) for name, path in model_paths.items()}

    # Verarbeitung der Audiodateien
    max_threads = 8
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        for dirpath, dirnames, filenames in os.walk(root_directory_path):
            for filename in filenames:
                if filename.lower().endswith((".wav", ".WAV")):
                    # Erstellen des vollständigen Dateipfades
                    file_path = os.path.join(dirpath, filename)
                    # Übergeben des vollständigen Dateipfades anstatt des Dateinamens und des Verzeichnispfades
                    executor.submit(process_file, file_path, models, collection)

    print("Alle Dateien wurden verarbeitet.")
