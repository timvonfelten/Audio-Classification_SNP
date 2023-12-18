# Audio Classification SNP

This repository contains a Deep Learning project focused on the classification of audio data. It includes various scripts for data preprocessing, model training, prediction, and hyperparameter tuning. Additionally, a pipeline script is provided for executing the entire processing flow with data storage in MongoDB.



This project is based on the repository [Audio-Classification](https://github.com/seth814/Audio-Classification) by Seth Adams.

## Getting Started

To get started, follow these steps:

1. Create a conda environment with Python 3.9:
         ```
         conda create -n audio python=3.9
         ```

2. Activate the conda environment:
         ```
         conda activate audio
         ```

3. Install the required packages:
         ```
         pip install -r requirements.txt
         ```

4. Run the scripts as needed. Pass the function arguments when executing the scripts, for example: `python clean.py --src_root audiofiles/wav`.

## Training Model

Use the following scripts to develop your models and test them. Select a samplerate and desired delta time which remain consistent throughout the process.

### 1. Audio Preprocessing

The `clean.py` script can be used to preview the signal envelope at a threshold and remove low magnitude data. Uncommenting `split_wavs` will create a clean directory with downsampled mono audio split by delta time.

- **clean.py**
        - **Purpose**: Preprocesses audio data.
        - **Inputs**:
                - `src_root`: Source directory of raw audio files.
                - `dst_root`: Destination directory for preprocessed files.
                - `delta_time`: Time frame for processing.
                - `sr`: Sampling rate.
                - `fn`: Feature normalization option.
                - `threshold`: Threshold for filtering.

### 2. Model Parameters

In `modles.py` various model parameters depending on the selected model. These parameters can be adjusted to customize the model architecture and behavior.

- **models.py**
        - **Purpose**: Contains model architectures (Conv1D, Conv2D, LSTM).
        - **Inputs**: Various model parameters depending on the selected model.

### 3. Model Training

In `train.py` the model training is done using the specified inputs in `modles.py`.

- **train.py**
        - **Purpose**: Trains the model on preprocessed audio data.
        - **Inputs**:
                - `model_type`: Type of the model (Conv1D, Conv2D, LSTM).
                - `src_root`: Directory of preprocessed data.
                - `batch_size`: Size of each training batch.
                - `delta_time`: Time frame for processing.
                - `sample_rate`: Sampling rate.

Use `PlotHistory.ipynb` to visualize the training process.

### 4. Prediction

In `predict.py` a trained modell can be tested doing inference.

- **predict.py**
        - **Purpose**: Performs inference using a trained model.
        - **Inputs**:
                - `model_fn`: Filename of the trained model.
                - `pred_fn`: Filename for saving predictions.
                - `src_dir`: Directory of data for prediction.
                - `dt`: Delta time for processing.
                - `sr`: Sampling rate.
                - `threshold`: Threshold for filtering.

### 5. Hyperparameter Tuning

The following scripts are used for hyperparameter tuning using random search. This step is optionl but can be useful in finding the best parameters.

- `hyperparametertuning_conv1d.py`
- `hyperparametertuning_conv2d.py`
- `hyperparametertuning_lstm.py`

- **Inputs**:
        - `objective`: Objective function for optimization.
        - `max_trials`: Maximum number of trials.
        - `executions_per_trial`: Number of executions per trial.
        - `directory`: Directory for saving results.
        - `project_name`: Name of the tuning project.
- **Parameters for Optimization**: `n_mels`, `n_fft`, `win_length`, `hop_length`, `dropout_rate`, `batch_size`.

## Run Pipeline
Use the following script to classify your data with pretrained models and save the results in MongoDB:

**pipeline/00_Pipeline.py**

- **Purpose**: Script to execute the entire processing pipeline with storage in MongoDB.

- **MongoDB Connection Information**:
    - `MONGO_URI`: MongoDB connection URI.
    - `DB_NAME`: Name of the MongoDB database.

- **Collection Names**:
    - `AUDIO_COLLECTION_NAME`: Name of the collection to store audio classification results.
    - `AGGREGATED_COLLECTION_NAME`: Name of the collection to store aggregated audio data.
    - `AGGREGATED_PER_DAY_COLLECTION_NAME`: Name of the collection to store aggregated daily audio data.

- **Model Paths**:
    - `MODEL_PATHS`: Dictionary containing the paths to the pre-trained models for different audio classes.

- **Directory Path**:
    - `DIRECTORY_PATH`: Path to the directory containing the audio files.

- **Confidence Threshold**:
    - `CONFIDENCE_THRESHOLD`: Threshold value for the confidence score of audio classification.

- **Maximum Threads**:
    - `MAX_THREADS`: Maximum number of threads to use for processing the audio files.

### Contact

For any questions or contributions, please open an issue in the repository.
