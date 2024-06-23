# Define paths
dataset_path = 'Datasets/RAVDESS/'  # RAVDESS Video Path
landmarks_path = 'Datasets/Original_FacialLandmark/'  # RAVDESS Landmarks Path
updated_landmarks_path = 'Datasets/Updated_FacialLandmark/'  # only relevant info
audios_dataset_path = 'Datasets/WAV/'
trim_audios_path = 'Datasets/WAV_TRIMMED/'
chunked_audios_path = 'Datasets/WAV_CHUNKS/'

csv_path = 'Datasets/CSV/'
logs_path = 'Logs/'
other_path = 'OTHER/'
models_path = 'Models/'

frames_path = 'Generated/Frames/'  # Created Images Path
flows_path = 'Generated/Flows/'  # Created Optic Flow frame Path
audios_path = 'Generated/Audios/'

test_frames_path = 'Generated/Test/Frames/'
test_flows_path = 'Generated/Test/Flows/'
test_audios_path = 'Generated/Test/Audios/'

RAVDESS_SAMPLE_RATE = 48000
VIDEO_FORMAT_CODECS = {'mp4': 'avc1', 'avi': 'MJPG'}

MAX_WIDTH = MAX_HEIGHT = 300
IMG_HEIGHT = IMG_WIDTH = 224
DATASET_TOTAL_ELEMENTS = 158262
DATASET_NUM_CLASSES = 8

MAX_SPEECH_SEQUENCE_LENGTH = 176253

EMOTIONS_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry',
                   'fearful', 'disgust', 'surprised']

EMOTIONS_LABELS_SORTED = ['angry', 'calm', 'disgust', 'fearful',
                          'happy', 'neutral', 'sad', 'surprised']