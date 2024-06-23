import constants as const
import sys
import time
# from constants import *
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TODO: correct difformity bw this emotions labels and constant one
EMOTIONS_LABELS = ['', 'neutral', 'calm', 'happy', 'sad', 'angry',
                   'fearful', 'disgust', 'surprised']

IMG_HEIGHT = const.IMG_HEIGHT
IMG_WIDTH = const.IMG_WIDTH
MAX_WIDTH = const.MAX_WIDTH
MAX_HEIGHT = const.MAX_HEIGHT

VIDEO_FORMAT_CODECS = const.VIDEO_FORMAT_CODECS


def timer():
    '''
    Starts/stops a timer
    '''
    t = time.time()
    return t


def trunc(values, decs=0):
    '''
    Truncate decimal values
    '''
    return np.trunc(values * 10 ** decs) / (10 ** decs)


def system_info():
    '''
    Print some useful info about python and environment
    '''
    # Print python infos
    win_v = sys.getwindowsversion()
    print(f'''
    Windows:
        Version: {win_v.major}.{win_v.minor} build {win_v.build}
    Python:
        Version: {sys.version}
        Path: {sys.executable}
    ''')


def modules_info():
    '''
    Print some useful info about modules
    '''
    print(f'''
    OpenCV:
        Version: {cv2.__version__}
    Tensorflow:
        Version: {tf.__version__}
    ''')


def video_info(vidcap):
    print(f'''
    Frame Width: {vidcap.get(cv2.CAP_PROP_FRAME_WIDTH):n}
    Frame Height: {vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT):n}
    Frame Rate: {vidcap.get(cv2.CAP_PROP_FPS):n}
    # Current position of video  (ms): {vidcap.get(cv2.CAP_PROP_POS_MSEC):n}
    Numbers of frame: {vidcap.get(cv2.CAP_PROP_FRAME_COUNT):n}
    ''')


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


# Remove extension from filename

def remove_extension(filename):
    if (not isinstance(filename, Path)):
        filename = Path(filename)

    return filename.stem


# Print human readable info of video based on RAVDESS dataset naming schema

def filename_info(filename):
    '''
    Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
    Vocal channel (01 = speech, 02 = song).
    Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry,\
    06 = fearful, 07 = disgust, 08 = surprised).
    Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong\
    intensity for the 'neutral' emotion.
    Statement (01 = "Kids are talking by the door",\
    02 = "Dogs are sitting by the door").
    Repetition (01 = 1st repetition, 02 = 2nd repetition).
    Actor (01 to 24. Odd numbered actors are male, even numbered actors are\
    female).
    '''
    frame_number = None

    filename = filename.split('/')[-1]  # take last path and remove full path
    filename = remove_extension(filename)  # remove filename extension
    # if filename is a image the last number is the frame number
    split_frame = filename.split('_')

    split_var = split_frame[0].split('-')
    if len(split_frame) == 2:
        frame_number = int(split_frame[1])




    modality_l = ['', 'Full-AV', 'Video-only', 'Audio-only']
    channel_l = ['', 'Speech', 'Song']
    emotion_l = ['', 'Neutral', 'Calm', 'Happy', 'Sad', 'Angry',
                 'Fearful', 'Disgust', 'Surprised']
    intensity_l = ['', 'Normal', 'Strong']
    statement_l = ['', 'Kids are talking by the door',
                   'Dogs are sitting by the door']
    repetition_l = ['', '1st repetition', '2nd repetition']
    gender_l = ['Female', 'Male']


    actor = int(split_var[6])
    # gender = 'female' if (int(actor)%2 == 0) else 'male'
    modality = int(split_var[0])
    channel = int(split_var[1])
    intensity = int(split_var[3])
    statement = int(split_var[4])
    repetition = int(split_var[5])
    emotion = int(split_var[2])

    print(f'Modality: {modality_l[modality]}')
    print(f'Channel: {channel_l[channel]}')
    print(f'Emotion: {emotion_l[emotion]}')
    print(f'Intensity: {intensity_l[intensity]}')
    print(f'Statement: {statement_l[statement]}')
    print(f'Repetition: {repetition_l[repetition]}')
    print(f'Actor: {actor} ({gender_l[int(actor)%2]})')

    if (frame_number):
        print(f'Frame Number: {frame_number}')


# Print a human readable size of a file
# Source:
# https://web.archive.org/web/20111010015624/http://blogmag.net/blog/read/38/Print_human_readable_file_size

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


# Estimate space considering only spoken part of video (avg 60% of video)

def estimated_space_occupation(number_of_videos, size=str(128)):
    '''
    Estimate disk space occupation of generating frames for different videos.
        Args:
            number_of_videos: Number of videos to estimate
            size = width/length of the frames to generate (size x size)

        Returns:
            Esteem of the size on disk
    '''

    avg_frames_per_video = 60
    avg_img_size = {'64': 3000, '128': 7000, '300': 26000}  # in kB
    frames_generated = avg_frames_per_video * number_of_videos
    assert size in avg_img_size, 'Size not included'
    esteem = frames_generated * avg_img_size[size]

    return [frames_generated, sizeof_fmt(esteem)]


# Convert audio index to video frame

def index_to_frame(index, sample_rate=48000, frame_rate=29.97):
    a = index / sample_rate
    b = a * frame_rate

    frame = int(np.ceil(b))
    return frame


# Convert video filename to audio filename

def video_to_audio_filename(full_path):
    # parent_folders = full_path.parent
    # actor_folder = parent_folders.relative_to(parent_folders.parent)

    name = full_path.stem
    name = name.replace('1', '3', 1)  # audio only is 03
    name = f'{name}.wav'

    audio_filename = Path(const.audios_dataset_path, get_class_string(name),
                          name)
    return audio_filename



def get_class(fname):
    """ Retrieve the name of the class given a filename.

    Args:
      fname: Name of the file in the RAVDESS dataset.

    Returns:
      Class that the file belongs to.
    """
    return fname.split('-')[2]


def get_class_string(fname):
    """ Retrieve the name of the class given a filename as text.

    Args:
      fname: Name of the file in the RAVDESS dataset.

    Returns:
      Class that the file belongs to.
    """
    target_class = int(get_class(fname))
    return EMOTIONS_LABELS[target_class]


def get_actor(fname):
    """ Retrieve the actor of the given filename.

    Args:
      fname: Name of the file in the RAVDESS dataset.

    Returns:
      Actor
    """
    return fname.split('-')[6]


def get_gender(fname):
    """ Retrieve the gender of the given filename.

    Args:
      fname: Name of the file in the RAVDESS dataset.

    Returns:
      Gender
    """
    gender = int(get_actor(fname)) % 2
    gender_str = 'male' if (gender == 1) else 'female'
    return [gender, gender_str]


def get_full_path(row):
    # extract full path from a video entry in pandas df
    path = row.path
    filename = row.filename

    full_path = str(Path(path, filename))

    return full_path


# https://gist.github.com/soply/f3eec2e79c165e39c9d540e916142ae1

def show_images(images, cols=1, titles=None, to_gray=False):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """

    if not isinstance(images, list):
        raise TypeError('Type should be list')

    if (titles is not None) and (len(titles) != len(images)):
        raise ValueError('Title should be None or of length equal to images')


    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(int(np.ceil(n_images / float(cols))), cols, n + 1)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (to_gray):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            plt.gray()
        # if image.ndim == 2:
        #     plt.gray()


        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def show_plots(plots, cols=1, titles=None):
    """Display a list of plots in a single figure with matplotlib.

    Parameters
    ---------
    plots: List of plots compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each plots. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(plots) == len(titles)))
    n_plots = len(plots)

    if titles is None:
        titles = ['Plot (%d)' % i for i in range(1, n_plots + 1)]

    fig = plt.figure()
    for n, (plot, title) in enumerate(zip(plots, titles)):

        a = fig.add_subplot(int(np.ceil(n_plots / float(cols))), cols, n + 1)

        plt.show(plot)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_plots)
    plt.show()


def show_video(v_path):

    if not isinstance(v_path, (str, type(Path('.')))):
        raise TypeError('Type should be str or pathlib Path')

    if not Path(v_path).is_file():
        raise FileNotFoundError('File not found')

    v_path = str(v_path)
    vidcap = cv2.VideoCapture(v_path)

    # utils.video_info(vidcap)

    while (True):
        ret, frame = vidcap.read()
        if not ret:
            print('No frames grabbed!')
            break

        cv2.imshow('Frame', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:  # ESC
            break

    vidcap.release()
    cv2.destroyAllWindows()


def show_waveform(audio, title='', ax=None):

    if (ax is None):
        fig, ax = plt.subplots()
    timescale = np.arange(audio.shape[0])
    ax.plot(timescale, audio)
    ax.set_xlim([0, audio.shape[0]])
    ax.set_title(title)


def video_search(df, modalities=None, channels=None, emotions=None,
                 intensities=None, statements=None, repetitions=None,
                 actors=None):

    '''
        Returns an array of videos with the input characteristics


        Parameters
        ----------
        modality: list, optional
            Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
        channel: list, optional
            Vocal channel (01 = speech, 02 = song)
        emotion: list, optional
            (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry,\
            06 = fearful, 07 = disgust, 08 = surprised)
        intensity: list, optional
            Emotional intensity (01 = normal, 02 = strong).\
            NOTE: There is no strong intensity for the 'neutral' emotion
        statement: list, optional
            Statement (01 = "Kids are talking by the door",\
            02 = "Dogs are sitting by the door")
        repetition: list, optional
            Repetition (01 = 1st repetition, 02 = 2nd repetition)
        actor: list, optional
            Actor (01 to 24. Odd numbered actors are male,\
            even numbered actors are female)


    '''

    c1 = c2 = c3 = c4 = c5 = c6 = c7 = True

    # c1 = (df.modality == modality) if (modality) else True
    # c2 = (df.channel == channel) if (channel) else True
    # c3 = (df.emotion == emotion) if (emotion) else True
    # c4 = (df.intensity == intensity) if (intensity) else True
    # c5 = (df.statement == statement) if (statement) else True
    # c6 = (df.repetition == repetition) if (repetition) else True
    # c7 = (df.actor == actors) if (actors) else True

    if modalities:
        c1 = False
        for modality in modalities:
            c1 |= (df.modality == modality)  # bit-wise OR in series

    if channels:
        c2 = False
        for channel in channels:
            c2 |= (df.channel == channel)  # bit-wise OR in series

    if emotions:
        c3 = False
        for emotion in emotions:
            c3 |= (df.emotion == emotion)  # bit-wise OR in series

    if intensities:
        c4 = False
        for intensity in intensities:
            c4 |= (df.intensity == intensity)  # bit-wise OR in series

    if statements:
        c5 = False
        for statement in statements:
            c5 |= (df.statement == statement)  # bit-wise OR in series

    if repetitions:
        c6 = False
        for repetition in repetitions:
            c6 |= (df.repetition == repetition)  # bit-wise OR in series

    if actors:
        c7 = False
        for actor in actors:
            c7 |= (df.actor == actor)  # bit-wise OR in series

    c0 = c1 & c2 & c3 & c4 & c5 & c6 & c7

    results = df.loc[c0] if (c0 is not True) else df

    return results

# TENSOR AND NUMPY UTIL FUNCTIONS


def extend_np_array(X, axis=-1):
    '''
    Add a new axis to the np array

    Args:
        X: np array
        axis: default -1

    Returns:
        Y: X with added axis
    '''

    Y = np.expand_dims(a=X, axis=axis)

    return Y


def extend_tensor(X, axis=-1, name=None):
    '''
    Add a new axis to the tensor

    Args:
        X: tensor
        axis: default -1

    Returns:
        Y: X with added axis
    '''

    Y = tf.expand_dims(X, axis, name)

    return Y


def squeeze_np_array(X, axis=-1):
    '''
    Squeeze a np array axis

    Args:
        X: np array
        axis

    Returns:
        Y: X with removed axis
    '''

    Y = np.squeeze(X, axis=axis)

    return Y


def squeeze_tensor(X, axis=-1, name=None):
    '''
    Squeeze tensor axis

    Args:
        X: tensor
        axis

    Returns:
        Y: X with removed axis
    '''

    Y = tf.squeeze(X, axis=axis, name=name)

    return Y


def tensor_to_numpy(X):

    # if not isinstance(waveform, np.ndarray):
    if tf.is_tensor(X):
        try:
            Y = X.numpy()
        except AttributeError:
            print(f'Error: no numpy attribute: {type(X)}')
            return
    else:
        print(f'input is not a tensor: {type(X)}')
        return

    return Y


def get_video_frames(vidcap):

    try:
        return int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    except AttributeError:
        print('vidcap has no attribute CAP_PROP_FRAME_COUNT')


def get_video_fps(vidcap):

    try:
        return int(vidcap.get(cv2.CAP_PROP_FPS))
    except AttributeError:
        print('vidcap has no attribute CAP_PROP_FPS')


def get_different_indexes_matrix(A, B):

    A_shape = A.shape
    B_shape = B.shape

    assert A_shape == B_shape, f'Different shapes: A({A_shape}), B({B_shape})'


    x = A == B
    x_inverted = np.invert(x)
    a = np.where(x_inverted)
    n_dim = len(a)
    n_elem = len(a[0])
    diff_arr = []

    for x in range(n_elem):
        i = 0
        str = []
        while (i < n_dim):
            str.append(a[i][x])
            i += 1

        diff_arr.append(tuple(str))

    return diff_arr


def print_different_elements_matrix(A, B, elements=False,
                                    differences=False, max_difference=True):

    res = get_different_indexes_matrix(A, B)

    diff_arr = []

    for idx in res:

        A_el = A[idx]
        B_el = B[idx]

        el_str = f'[{idx}]: A:{A_el}, B: {B_el}' if elements else ''

        diff = abs(A_el - B_el)

        diff_str = f', diff: {diff:.2f}' if differences else ''
        diff_arr.append(diff)

        full_str = f'{el_str}{diff_str}'
        print(full_str) if full_str else ''

    if (diff_arr and max_difference):
        print()
        print(f'max_diff: {np.max(diff_arr)}')


def video_from_images(input_path, output_path, size=(IMG_HEIGHT, IMG_WIDTH),
                      color=True, fps=30, view=False, save=True):
    '''
    Create a video from a series of images
    '''
    images_paths = input_path.glob('*.jpg')
    images_paths = [str(img) for img in images_paths]

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FORMAT_CODECS['mp4'])  # mp4 h264
    if save:
        out = cv2.VideoWriter(str(output_path), fourcc, fps, size, color)

    for image_path in images_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, size)
        if view:
            cv2.imshow("Frame", image)
            key = cv2.waitKey(int(1000/fps))
            if key == 27:
                break

        if save:
            out.write(image)
    if save:
        out.release()
    cv2.destroyAllWindows()


def calc_bounding_box(data, frame_index):
    '''
        Create the bounding box from pandas df and frame index
    '''
    idx = frame_index - 1

    # Take min and max for this frame
    x_max = int(data.iloc[idx].max_x)
    x_min = int(data.iloc[idx].min_x)
    y_max = int(data.iloc[idx].max_y)
    y_min = int(data.iloc[idx].min_y)

    # Calculate the width and height of the bounding box
    w = max((x_max - x_min), MAX_WIDTH)
    h = max((y_max - y_min), MAX_HEIGHT)

    # Calculate the maximum to make a squared bounding box
    m = max(w, h)


    return {
        'box': [x_min, y_min, m, m]
    }