# *_*coding:utf-8 *_*
import os

DATA_DIR = {
	'Track1_English': 'datasets/MC-EIU_Track1_English_Processed',
	'Track1_Mandarin': 'datasets/MC-EIU_Track1_Mandarin_Processed',
	'Track2_English': 'datasets/MC-EIU/Track2/English',
	'Track2_Mandarin': 'datasets/MC-EIU/Track2/Mandarin',
}

DATA_PROCESSED_DIR = {
	'Track1_English': 'datasets/MC-EIU_Track1_English_Processed',
	'Track1_Mandarin': 'datasets/MC-EIU_Track1_Mandarin_Processed',
	'Track2_English': 'datasets/MC-EIU/Processed/Track2/English',
	'Track2_Mandarin': 'datasets/MC-EIU/Processed/Track2/Mandarin',
}

PATH_TO_RAW_AUDIO = {
	'Track1_English': os.path.join(DATA_PROCESSED_DIR['Track1_English'], 'Audios'),
	'Track1_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track1_Mandarin'], 'Audios'),
	'Track2_English': os.path.join(DATA_PROCESSED_DIR['Track2_English'], 'Audios'),
	'Track2_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track2_Mandarin'], 'Audios'),
}

PATH_TO_RAW_FACE = {
	'Track1_English': os.path.join(DATA_PROCESSED_DIR['Track1_English'], 'Videos'),
	'Track1_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track1_Mandarin'], 'Videos'),
	'Track2_English': os.path.join(DATA_PROCESSED_DIR['Track2_English'], 'Videos'),
	'Track2_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track2_Mandarin'], 'Videos'),
}

PATH_TO_OPENFACE_ALL = {
	'Track1_English': os.path.join(DATA_PROCESSED_DIR['Track1_English'], 'Features', 'openface_all'),
	'Track1_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track1_Mandarin'], 'Features', 'openface_all'),
	'Track2_English': os.path.join(DATA_PROCESSED_DIR['Track2_English'], 'Features', 'openface_all'),
	'Track2_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track2_Mandarin'], 'Features', 'openface_all'),
}

PATH_TO_TRANSCRIPTIONS = {
	'Track1_English': os.path.join(DATA_PROCESSED_DIR['Track1_English'], 'transcription.csv'),
	'Track1_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track1_Mandarin'], 'transcription.csv'),
	'Track2_English': os.path.join(DATA_DIR['Track2_English'], 'transcription.csv'),
	'Track2_Mandarin': os.path.join(DATA_DIR['Track2_Mandarin'], 'transcription.csv'),
}

PATH_TO_FEATURES = {
	'Track1_English': os.path.join(DATA_PROCESSED_DIR['Track1_English'], 'Features'),
	'Track1_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track1_Mandarin'], 'Features'),
	'Track2_English': os.path.join(DATA_PROCESSED_DIR['Track2_English'], 'Features'),
	'Track2_Mandarin': os.path.join(DATA_PROCESSED_DIR['Track2_Mandarin'], 'Features'),
}


PATH_TO_OPENFACE = "OpenFace/build/bin"
PATH_TO_PRETRAINED_MODELS = 'tools'
PATH_TO_OPENSMILE = os.path.join(PATH_TO_PRETRAINED_MODELS, r'opensmile-2.3.0')
PATH_TO_FFMPEG = os.path.join(PATH_TO_PRETRAINED_MODELS, r'ffmpeg-4.4.1-i686-static', r'ffmpeg')
PATH_TO_NOISE = os.path.join(PATH_TO_PRETRAINED_MODELS, r'musan', r'audio-select')

# SAVED_ROOT = os.path.join('./saved')
# DATA_DIR = os.path.join(SAVED_ROOT, 'data')
# MODEL_DIR = os.path.join(SAVED_ROOT, 'model')
# LOG_DIR = os.path.join(SAVED_ROOT, 'log')
# PREDICTION_DIR = os.path.join(SAVED_ROOT, 'prediction')
# FUSION_DIR = os.path.join(SAVED_ROOT, 'fusion')
# SUBMISSION_DIR = os.path.join(SAVED_ROOT, 'submission')
