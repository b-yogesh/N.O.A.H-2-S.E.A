#Inspired from https://github.com/birds-on-mars/birdsonearth for ZooHackathon

import pickle
import torch
import torch.nn as nn
import os
from librosa.core import load
import numpy as np
from scipy.io import wavfile
import resampy
from torch.nn.functional import relu, softmax

class Params:
    '''
    defines all parameters needed in training.py and model.py
    TRAINING
    n_epochs (int): number of training epochs
    batch_size (int):
    val_split (float): fraction of data kept for validation
    data_root (str): path to data
    n_max (int): max number of instances loaded for training
    weights (str): path to vggish weights .hdf5 file
    '''

    def __init__(self):

        # Data
        #TODO: add options for other formats
        self.data_format = 'wav'

        # Model
        self.n_bins = 64
        self.n_frames = 96
        self.n_classes = 3

        # Training
        self.n_epochs = 100
        self.batch_size = 512
        self.val_split = .2
        self.data_root = '../data/full_urbansounds_restructured'
        # if mel_spec_root directory exists it is used and preprocessing of data_root is skipped
        # otherwise mel specs are computed from data_root
        self.mel_spec_root = '../data/full_urbansounds_specs'
        self.n_max = None
        self.weights = 'models/vggish_audioset_weights_without_fc2.h5'

        # model zoo
        self.save_model = True
        self.model_zoo = 'models'
        self.name = 'urban'

        # computing device, can be 'cuda:<GPU index>' or 'cpu'
        self.device = 'cpu'

params = Params()

params.name = 'BirdsSongs'

# training
params.data_format = 'wav'
params.data_root = '/content/drive/My Drive/ZooHackathon/BirdSong'
params.mel_spec_root = '/content/drive/My Drive/ZooHackathon/BirdSong/processed'
params.n_epochs = 40
params.batch_size = 256
params.val_split = .2
params.save_model = True
params.n_max = 1000

class VGGish(nn.Module):

    def __init__(self, params):

        super(VGGish, self).__init__()

        self.n_bins = params.n_bins
        self.n_frames = params.n_frames
        self.out_dims = int(params.n_bins / 2**4 * params.n_frames / 2**4)
        self.n_classes = params.n_classes
        self.weights = params.weights
        self.model_zoo = params.model_zoo
        self.name = params.name

        # convolutional bottom part
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        # fully connected top part
        self.classifier = nn.Sequential(
            nn.Linear(self.out_dims*512, 1028),
            nn.ReLU(),
            nn.Linear(1028, 1028),
            nn.ReLU(),
            nn.Linear(1028, self.n_classes)
        )

    def forward(self, X):

        a = self.pool1(relu(self.conv1(X)))
        a = self.pool2(relu(self.conv2(a)))
        a = relu(self.conv3_1(a))
        a = relu(self.conv3_2(a))
        a = self.pool3(a)
        a = relu(self.conv4_1(a))
        a = relu(self.conv4_2(a))
        a = self.pool4(a)
        a = a.reshape((a.size(0), -1))
        a = self.classifier(a)
        a = softmax(a)
        return a

    def init_weights(self, file=None):
        '''
        laods pretrained weights from an .hdf5 file. File structure must match exactly.
        Args:
            file (string): path to .hdf5 file containing VGGish weights
        '''

        if file is not None:
            file = file
        else:
            file = self.weights

        # loading weights from file
        with h5.File(file, 'r') as f:

            conv1 = f['conv1']['conv1']
            kernels1 = torch.from_numpy(conv1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases1 = torch.from_numpy(conv1['bias:0'][()])
            conv2 = f['conv2']['conv2']
            kernels2 = torch.from_numpy(conv2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases2 = torch.from_numpy(conv2['bias:0'][()])
            conv3_1 = f['conv3']['conv3_1']['conv3']['conv3_1']
            kernels3_1 = torch.from_numpy(conv3_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_1 = torch.from_numpy(conv3_1['bias:0'][()])
            conv3_2 = f['conv3']['conv3_2']['conv3']['conv3_2']
            kernels3_2 = torch.from_numpy(conv3_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases3_2 = torch.from_numpy(conv3_2['bias:0'][()])
            conv4_1 = f['conv4']['conv4_1']['conv4']['conv4_1']
            kernels4_1 = torch.from_numpy(conv4_1['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_1 = torch.from_numpy(conv4_1['bias:0'][()])
            conv4_2 = f['conv4']['conv4_2']['conv4']['conv4_2']
            kernels4_2 = torch.from_numpy(conv4_2['kernel:0'][()].transpose(3, 2, 1, 0))
            biases4_2 = torch.from_numpy(conv4_2['bias:0'][()])

            # assigning weights to layers
            self.conv1.weight.data = kernels1
            self.conv1.bias.data = biases1
            self.conv2.weight.data = kernels2
            self.conv2.bias.data = biases2
            self.conv3_1.weight.data = kernels3_1
            self.conv3_1.bias.data = biases3_1
            self.conv3_2.weight.data = kernels3_2
            self.conv3_2.bias.data = biases3_2
            self.conv4_1.weight.data = kernels4_1
            self.conv4_1.bias.data = biases4_1
            self.conv4_2.weight.data = kernels4_2
            self.conv4_2.bias.data = biases4_2

    def freeze_bottom(self):
        '''
        freezes the convolutional bottom part of the model.
        '''
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False

    def save_weights(self):
        torch.save(self.state_dict(),
                    '/content/drive/My Drive/ZooHackathon/BirdSong/' + self.name+'.pt')
        return

def preprocess(src, dst):
    y, sr = load(src)
    y *= 32768
    y = y.astype(np.int16)
    wavfile.write(dst, rate=22050, data=y)

def wavfile_to_examples(wav_file):
  """Convenience wrapper around waveform_to_examples() for a common WAV format.
  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.
  Returns:
    See waveform_to_examples.
  """
  sr, wav_data = wavfile.read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  return waveform_to_examples(samples, sr)

def waveform_to_examples(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.
  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.
  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate !=  SAMPLE_RATE:
    data = resampy.resample(data, sample_rate,  SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = log_mel_spectrogram(
      data,
      audio_sample_rate= SAMPLE_RATE,
      log_offset= LOG_OFFSET,
      window_length_secs= STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs= STFT_HOP_LENGTH_SECONDS,
      num_mel_bins= NUM_MEL_BINS,
      lower_edge_hertz= MEL_MIN_HZ,
      upper_edge_hertz= MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 /  STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
       EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
       EXAMPLE_HOP_SECONDS * features_sample_rate))

  log_mel_examples = frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples

# Architectural constants.
NUM_FRAMES = 96  # Frames in input mel-spectrogram patch.
NUM_BANDS = 64  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.

# Hyperparameters used in feature and example generation.
SAMPLE_RATE = 16000
STFT_WINDOW_LENGTH_SECONDS = 0.025
STFT_HOP_LENGTH_SECONDS = 0.010
NUM_MEL_BINS = NUM_BANDS
MEL_MIN_HZ = 125
MEL_MAX_HZ = 7500
LOG_OFFSET = 0.01  # Offset used for stabilized log of input mel-spectrogram.
EXAMPLE_WINDOW_SECONDS = 0.96  # Each example contains 96 10ms frames
EXAMPLE_HOP_SECONDS = 0.96     # with zero overlap.

# Parameters used for embedding postprocessing.
PCA_EIGEN_VECTORS_NAME = 'pca_eigen_vectors'
PCA_MEANS_NAME = 'pca_means'
QUANTIZE_MIN_VAL = -2.0
QUANTIZE_MAX_VAL = +2.0

# Hyperparameters used in training.
INIT_STDDEV = 0.01  # Standard deviation used to initialize weights.
LEARNING_RATE = 1e-4  # Learning rate for the Adam optimizer.
ADAM_EPSILON = 1e-8  # Epsilon for the Adam optimizer.

# Names of ops, tensors, and features.
INPUT_OP_NAME = 'vggish/input_features'
INPUT_TENSOR_NAME = INPUT_OP_NAME + ':0'
OUTPUT_OP_NAME = 'vggish/embedding'
OUTPUT_TENSOR_NAME = OUTPUT_OP_NAME + ':0'
AUDIO_EMBEDDING_FEATURE_NAME = 'audio_embedding'

_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def log_mel_spectrogram(data,
                        audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
  """Convert waveform to a log magnitude mel-frequency spectrogram.
  Args:
    data: 1D np.array of waveform data.
    audio_sample_rate: The sampling rate of data.
    log_offset: Add this to values when taking log to avoid -Infs.
    window_length_secs: Duration of each window to analyze.
    hop_length_secs: Advance between successive analysis windows.
    **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.
  Returns:
    2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
    magnitudes for successive frames.
  """
  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, **kwargs))
  return np.log(mel_spectrogram + log_offset)

def frame(data, window_length, hop_length):
  """Convert array into a sequence of successive possibly overlapping frames.
  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.
  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.
  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
  num_samples = data.shape[0]
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  shape = (num_frames, window_length) + data.shape[1:]
  strides = (data.strides[0] * hop_length,) + data.strides
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)

def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.
  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.
  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

def periodic_hann(window_length):
  """Calculate a "periodic" Hann window.
  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.
  Args:
    window_length: The number of points in the returned window.
  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))

def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
  """Return a matrix that can post-multiply spectrogram rows to make mel.
  Returns a np.array matrix A that can be used to post-multiply a matrix S of
  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
  "mel spectrogram" M of frames x num_mel_bins.  M = S A.
  The classic HTK algorithm exploits the complementarity of adjacent mel bands
  to multiply each FFT bin by only one mel weight, then add it, with positive
  and negative signs, to the two adjacent mel bands to which that bin
  contributes.  Here, by expressing this operation as a matrix multiply, we go
  from num_fft multiplies per frame (plus around 2*num_fft adds) to around
  num_fft^2 multiplies and adds.  However, because these are all presumably
  accomplished in a single call to np.dot(), it's not clear which approach is
  faster in Python.  The matrix multiplication has the attraction of being more
  general and flexible, and much easier to read.
  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.  This is
      the number of columns in the output matrix.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    audio_sample_rate: Samples per second of the audio at the input to the
      spectrogram. We need this to figure out the actual frequencies for
      each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.  This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.
  Returns:
    An np.array with shape (num_spectrogram_bins, num_mel_bins).
  Raises:
    ValueError: if frequency edges are incorrectly ordered or out of range.
  """
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  # HTK excludes the spectrogram DC bin; make sure it always gets a zero
  # coefficient.
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix

def hertz_to_mel(frequencies_hertz):
  """Convert frequencies to mel scale using HTK formula.
  Args:
    frequencies_hertz: Scalar or np.array of frequencies in hertz.
  Returns:
    Object of same size as frequencies_hertz containing corresponding values
    on the mel scale.
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def load_model_with(params, path, ptPath):
    print('loading model')
    # load class labels
    with open(path, 'rb') as f:
        labels = pickle.load(f)
    # init network and load weights
    params.n_classes = len(labels)
    device = torch.device(params.device)
    net = VGGish(params)
    new_top = torch.nn.Linear(net.out_dims*512, net.n_classes)
    net.classifier = new_top
    net.load_state_dict(torch.load(ptPath,
                        map_location=device))
    net.to(device)
    net.eval()
    print('model for labels {} is ready'.format(labels))
    return net, labels


def predict(net, labels, files, params):
    print('starting inference')
    device = torch.device(params.device)
    predictions = []
    probs = []
    for i, file in enumerate(files):
        filename = os.path.splitext(os.path.basename(file))[0]
        processed = filename + '_proc.wav'
        preprocess(file, processed)
        data = wavfile_to_examples(processed)
        data = torch.from_numpy(data).unsqueeze(1).float()
        data = data.to(device)
        net.to(device)
        out = net(data)
        # # for each spectrogram/row index of max probability
        # pred = np.argmax(out.detach().cpu().numpy(), axis=1)
        # # find most frequent index over all spectrograms
        # consensus = np.bincount(pred).argmax()
        # print('file {} sounds like a {} to me'.format(i, labels[consensus]))
        # mean probabilities for each col/class over all spectrograms
        mean_probs = np.mean(out.detach().cpu().numpy(), axis=0)
        # find index of max mean_probs
        idx = np.argmax(mean_probs)
        print('file {} sounds like a {} to me'.format(i, labels[idx]))
        print('my guesses are: ')
        for j, label in enumerate(labels):
            print('{0}: {1:.04f}'.format(label, mean_probs[j]))
        # predictions.append(labels[consensus])
        predictions.append(labels[idx])
        probs.append(mean_probs)
        os.remove(processed)
    return predictions, probs

def run_prediction(input):
    net, labels = load_model_with(params, '/content/drive/My Drive/ZooHackathon/BirdSong/BirdSongClassifier.pkl', '/content/drive/My Drive/ZooHackathon/BirdSong/BirdsSongs.pt')
    predictions, probs = predict(net, labels, input, params)
    return predictions, probs



'''
Example on how to use

input = ['/content/drive/My Drive/ZooHackathon/BirdSong/Rusty_Blackbird/544790.wav']

predictions, probs = run_prediction(input)
'''

