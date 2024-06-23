import tensorflow as tf
import utils
import scipy
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import wave
from pathlib import Path

FRAME_LENGTH = FFT_LENGTH = 2048  # Windows size
FRAME_STEP = 512
N_MELS = 128
N_MFCC = 30
SAMPLE_RATE = 48000


def calc_spec_shape(frame_length=FRAME_LENGTH, sr=SAMPLE_RATE,
                    samples=SAMPLE_RATE,
                    frame_step=FRAME_STEP,
                    n_mels=N_MELS, n_mfcc=N_MFCC):

    print('Simulating shapes and info ...')
    win_ms = (frame_length / sr) * 10e3
    step_ms = (frame_step / sr) * 10e3

    print(f'Win size is {int(win_ms)} ms, sliding to right {int(step_ms)} ms')

    f_bins = int((frame_length / 2) + 1)
    frames = int(((samples - frame_length) / frame_step) + 1)

    spec = (f_bins, frames)
    print(f'Spectrogram shape:{spec}  [frequency_bins, frames]')

    mel_spec = (n_mels, frames)
    print(f'Mel Spectrogram shape:{mel_spec}  [n_mels, frames]')

    if (n_mfcc > n_mels):
        raise ValueError(
            f'It must be: n_mfcc <= n_mels \nmfcc={n_mfcc}, n_mels={n_mels}'
        )

    mfccs = (n_mfcc, frames)
    print(f'MFCC shape:{mfccs}  [n_mfcc, frames]')


def get_spectrogram(waveform, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                    fft_length=FFT_LENGTH, transpose_res=True,
                    extend_res=True):

    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        signals=waveform,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length
    )

    # Obtain the magnitude of the STFT, neglecting the phase
    spectrogram = tf.abs(spectrogram)

    if transpose_res:
        # transpose, so that the time is represented on the x-axis (columns).
        spectrogram = tf.transpose(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    if extend_res:
        spectrogram = utils.extend_tensor(spectrogram)

    return spectrogram


def get_tf_mel_spectrogram(S, num_mel_bins=N_MELS, num_spectrogram_bins=None,
                           sample_rate=SAMPLE_RATE, fmin=0, fmax=None,
                           transpose_res=True):

    if (fmax is None):
        fmax = float(sample_rate) / 2

    if (num_spectrogram_bins is None):
        num_spectrogram_bins = S.shape[0]

    A = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_mel_bins,
        num_spectrogram_bins=num_spectrogram_bins,
        sample_rate=sample_rate,
        lower_edge_hertz=fmin,
        upper_edge_hertz=fmax,
        dtype=tf.dtypes.float32,
        name=None
    )

    if transpose_res:
        S = tf.transpose(S)

    M = tf.matmul(S, A)

#    M = tf.tensordot(S, A, 1)
    if transpose_res:
        # transpose, so that the time is represented on the x-axis (columns).
        M = tf.transpose(M)


    return M


def get_tf_mfcc(S=None, name=None, n_mfccs=N_MFCC):

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms=S, name=name
    )
    return mfccs[..., :n_mfccs]


def get_tf_delta_mfcc(S=None, order=1, axis=-1, width=9, polyorder=None,
                      mode='interp', **kwargs):

    if (polyorder is None):
        polyorder = order
#    kwargs.setdefault("polyorder", order)
    delta_mfccs = scipy.signal.savgol_filter(
        S, width, deriv=order, polyorder=polyorder, axis=axis,
        mode=mode, **kwargs
    )

    return delta_mfccs


def get_librosa_spectrogram(waveform, n_fft=FFT_LENGTH,
                            win_length=FRAME_LENGTH,
                            hop_length=FRAME_STEP,
                            center=False):

    '''
    Use librosa stft to return a spectrogram

    Returns:
        spectrogram: spectrogram with shape (`height`, `width`, `channels`),
        where height: 1 + n_fft/2, width: n_frames (length of audio)
    '''

    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = librosa.stft(
        y=waveform,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=center
    )

    # Obtain the magnitude of the STFT.
    spectrogram = np.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = utils.extend_np_array(spectrogram)

    return spectrogram


def get_librosa_mel_spectrogram(S=None, y=None, sr=SAMPLE_RATE,
                                n_fft=FFT_LENGTH, hop_length=FRAME_STEP,
                                win_length=FRAME_LENGTH, n_mels=N_MELS,
                                center=False, htk=False, norm='slaney'):

    # tf_parameters -> center = True, htk = False, norm = None
    mel_spectrogram = librosa.feature.melspectrogram(
        S=S,
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        center=center,
        htk=htk,
        norm=norm,
    )

    return mel_spectrogram


def get_librosa_mfcc(S=None, y=None, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                     n_fft=FFT_LENGTH, hop_length=FRAME_STEP,
                     win_length=FRAME_LENGTH, n_mels=N_MELS,
                     center=False, htk=False, norm='ortho'):

    mfccs = librosa.feature.mfcc(
        S=S, y=y, n_mfcc=n_mfcc, sr=sr, hop_length=hop_length, n_fft=n_fft,
        win_length=win_length, n_mels=n_mels, center=center, htk=htk, norm=norm
    )

    return mfccs


def get_librosa_delta_mfcc(S=None, order=1):

    delta_mfccs = librosa.feature.delta(data=S, order=order)

    return delta_mfccs


def spec_to_image(spectrogram):

    # min-max scale to fit inside 8-bit range
    img = utils.scale_minmax(spectrogram, 0, 255).astype(np.uint8)
    # put low frequencies at the bottom in image
    img = np.flip(img, axis=0)
    # invert. make black==more energy
    img = 255 - img

    # img = exteextend_np_array(img)

    return img


def plot_spectrogram(spectrogram, ax):

    # Convert the frequencies to log scale and transpose to have time on X
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)


def plot_librosa_spectrogram(data, ax=None, sr=SAMPLE_RATE, n_fft=FFT_LENGTH,
                             hop_length=FRAME_STEP, win_length=FRAME_LENGTH,
                             x_axis='time', y_axis='linear', yticks=None,
                             shading='auto', title='A spectrogram'):

    if (len(data.shape) == 3):
        data = utils.squeeze_np_array(data)

    librosa.display.specshow(data,
                             sr=sr,
                             hop_length=hop_length,
                             n_fft=n_fft,
                             win_length=win_length,
                             x_axis=x_axis,
                             y_axis=y_axis,
                             shading=shading,
                             ax=ax)
    if (ax is not None):
        ax.set_title(title)


    if (yticks is not None):
        plt.yticks(yticks)
    else:
        plt.yticks()


def tf_power_to_dB(S, ref=1.0, amin=1e-10, top_db=80.0):

    am_spec = 10.0 * tf.experimental.numpy.log10(tf.math.maximum(amin, S))
    am_spec -= 10.0 * tf.experimental.numpy.log10(tf.math.maximum(amin, ref))
    am_spec = tf.math.maximum(am_spec, tf.math.reduce_max(am_spec) - top_db)

    return am_spec


def tf_amplitude_to_dB(S, amin=1e-05, ref=1.0, top_db=80):
    '''
    Tensorflow adaptation of librosa amplitude_to_db method

     Args:
        S: tensor spectrogram
        amin:
        ref:
        top_db

    Returns:
        am_spec: Tensor representing amplitude spectrogram
    '''

    magnitude = tf.abs(S)
    power = tf.square(magnitude)


    return tf_power_to_dB(power, ref=ref**2, amin=amin**2, top_db=top_db)


def batch_get_tf_mel_spectrogram(waveform, ref=1.0, amin=1e-05,
                                 top_db=80.0, batch=True):

    '''
    Create a mel spectrogram from a waveform

    Args:
        waveform: tensor with shape (None, int) or (int,).\
        See 'batch' flag for more.
        ref:
        amin:
        top_db:
        batch: if True, calculation are done considering a prepended shape\
        of None,
        to accomodate Tensorflow batches dimensions

    Returns:
        mel_spectrogram: mel spectrogram with shape (None, int, int, 1) or\
        (int, int, 1) according to batch flag
    '''


    # # Convert the waveform to a spectrogram via a STFT.
    # spectrogram = tf.signal.stft(
    #     signals=waveform,
    #     frame_length=FRAME_LENGTH,
    #     frame_step=FRAME_STEP,
    #     fft_length=FFT_LENGTH
    # )

    # # Obtain the magnitude of the STFT, neglecting the phase
    # spectrogram = tf.abs(spectrogram)

    spectrogram = get_spectrogram(
        waveform, transpose_res=False, extend_res=False
    )

    # spectrogram = tf.transpose(spectrogram, perm=[0, 2, 1])
#    print('Spectrogram:', spectrogram.shape)

    max_value = tf.reduce_max(spectrogram)
    ref = max_value if (ref == 'max') else ref

#    fmax = float(SAMPLE_RATE) / 2

    num_spec_bins = spectrogram.shape[2] if batch else spectrogram.shape[1]

    # A = tf.signal.linear_to_mel_weight_matrix(
    #     num_mel_bins=N_MELS,
    #     num_spectrogram_bins=num_spec_bins,
    #     sample_rate=SAMPLE_RATE,
    #     lower_edge_hertz=0,
    #     upper_edge_hertz=fmax,
    #     dtype=tf.dtypes.float32,
    #     name=None
    # )
    # [None,128,124]
    # S_T = tf.transpose(S, perm=[0, 1, 2])
    # print('S_t shape:', S_T.shape)

#    print('A shape:', A.shape)
#    M = tf.matmul(spectrogram, A)

    # M = tf.tensordot(S, A, 1)

#    perm_index = [0, 2, 1] if batch else [1, 0]

    # transpose, so that the time is represented on the x-axis (columns).
#    mel_spectrogram = tf.transpose(M, perm=perm_index)

    mel_spectrogram = get_tf_mel_spectrogram(
        spectrogram, num_spectrogram_bins=num_spec_bins, transpose_res=False
    )

    mel_power_spectrogram = tf_amplitude_to_dB(
        mel_spectrogram, amin, ref, top_db
    )

    # mfccs = get_tf_mfcc(mel_power_spectrogram)

# NOT WORKING WITH DELTA BECAUSE PASSING NP ARRAY
#    delta_mfccs = get_tf_delta_mfcc(mfccs)


    spec = utils.extend_tensor(mel_power_spectrogram)
    print('Out: ', spec.shape)
    return spec


def make_spec_ds(ds):
    return ds.map(
        map_func=lambda audio, label: (
            audio,
            # tf.py_function(
            #    func=get_dB_spectrogram, inp=[audio], Tout="float32"
            # ),
            batch_get_tf_mel_spectrogram(audio, ref='max'),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )


# check for faulty wav files

def compare_header_and_size(wav_filename):
    if not isinstance(wav_filename, (str, type(Path('.')))):
        raise TypeError('Input must be a string or Path')

    wav_filename = str(wav_filename)

    with wave.open(wav_filename, 'r') as fin:
        header_fsize = (
            fin.getnframes() * fin.getnchannels() * fin.getsampwidth() + 44
        )

    file_fsize = Path(wav_filename).stat().st_size

    return header_fsize != file_fsize