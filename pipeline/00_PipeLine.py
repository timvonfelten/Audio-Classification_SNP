from PL01_predict import process_audio_directory
from PL02_extractInfo import update_documents
from PL03_aggregate import aggregate_audio_data
from PL04_aggregateDay import aggregate_daily_audio_data

# MongoDB Verbindungsinformationen
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "ClassifiedAudioSNP"

# Sammlungsnamen
AUDIO_COLLECTION_NAME = "audio_classification"
AGGREGATED_COLLECTION_NAME = "aggregated_files"
AGGREGATED_PER_DAY_COLLECTION_NAME = "aggregated_files_per_day"

# Modellpfade
MODEL_PATHS = {
    'stille': "models\conv2d_stille_96.h5",
    'vogel': "models\conv2d_vogel_91.h5",
    'laerm': "models\conv2d_laerm_93.h5",
    'natur': "models\conv2d_natur_85.h5"
}

# Verzeichnispfad für Audio-Dateien
DIRECTORY_PATH = "D:\SNP_AudioDaten"

# Schwellenwert für die Konfidenz
CONFIDENCE_THRESHOLD = 0.9

MAX_THREADS = 8


def process_audio_pipeline():
    """
    This script processes an audio directory, performs audio classification using pre-trained models,
    and saves the results in a MongoDB database.

    Steps:
    1. Process audio directory and classify audio files using pre-trained models.
    2. Update documents in the MongoDB collection with the classification results.
    3. Aggregate audio data based on confidence threshold.
    4. Aggregate daily audio data.

    MongoDB Connection Information:
    - MONGO_URI: MongoDB connection URI.
    - DB_NAME: Name of the MongoDB database.

    Collection Names:
    - AUDIO_COLLECTION_NAME: Name of the collection to store audio classification results.
    - AGGREGATED_COLLECTION_NAME: Name of the collection to store aggregated audio data.
    - AGGREGATED_PER_DAY_COLLECTION_NAME: Name of the collection to store aggregated daily audio data.

    Model Paths:
    - MODEL_PATHS: Dictionary containing the paths to the pre-trained models for different audio classes.

    Directory Path:
    - DIRECTORY_PATH: Path to the directory containing the audio files.

    Confidence Threshold:
    - CONFIDENCE_THRESHOLD: Threshold value for the confidence score of audio classification.

    Maximum Threads:
    - MAX_THREADS: Maximum number of threads to use for processing the audio files.

    """

    print("Processing audio directory...")
    process_audio_directory(
        MONGO_URI,
        DB_NAME,
        AUDIO_COLLECTION_NAME,
        MODEL_PATHS,
        DIRECTORY_PATH
    )

    print("Updating documents...")
    update_documents(MONGO_URI, DB_NAME, AUDIO_COLLECTION_NAME)

    print("Aggregating audio data...")
    aggregate_audio_data(MONGO_URI, DB_NAME, AUDIO_COLLECTION_NAME, AGGREGATED_COLLECTION_NAME, CONFIDENCE_THRESHOLD)

    print("Aggregating daily audio data...")
    aggregate_daily_audio_data(MONGO_URI, DB_NAME, AGGREGATED_COLLECTION_NAME, AGGREGATED_PER_DAY_COLLECTION_NAME)


process_audio_pipeline()
