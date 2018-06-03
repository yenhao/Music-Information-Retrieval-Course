import os
import librosa
import librosa.display
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt


folder = '../datasets/Ballroom/'
train_folder = os.path.join(folder, 'BallroomData')
label_folder = os.path.join(folder, 'BallroomAnnotations')
song_types = [item for item in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder,item))]
# print(song_types)


def rms_odf(tfr):
    # tfr, f, t, Nfft = spec.Spectrogram(x, fr, fs, Hop, h)
    RMScurve = np.sum(tfr, axis = 0)
    ODF_RMS = np.zeros(RMScurve.shape)
    D = RMScurve[1:] - RMScurve[:-1]
    D[D<0] = 0
    ODF_RMS[1:] = D
    return RMScurve, ODF_RMS

def sf_odf(tfr):
    # tfr, f, t, Nfft = spec.Spectrogram(x, fr, fs, Hop, h)
    ODF = np.zeros(tfr.shape[1])
    A = tfr[:,1:] - tfr[:,:-1]
    A[A<0] = 0 # ReLU
    ODF[1:] = np.sum(A, axis = 0) # sum all value for different frequency
    return ODF

def plot_novoty_curve(RMScurve=None, ODF_RMS=None, ODF=None):

    if RMScurve.all() != None:
        plt.plot(range(RMScurve.shape[0]), RMScurve, label='RMScurve')
    if ODF_RMS.all() != None:
        plt.plot(range(RMScurve.shape[0]), ODF_RMS, label='ODF_RMS')
    if ODF.all() != None:
        plt.plot(range(RMScurve.shape[0]), ODF, label='ODF')

    plt.xlabel("Time")
    plt.ylabel("Onset")
    plt.legend()
    plt.show()
    plt.close()

def plot_spectrogram(D):
    librosa.display.specshow(librosa.amplitude_to_db(D, ref = np.max), y_axis = 'log', x_axis = 'time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":

    song_type = song_types[0]
    song_names = os.listdir(os.path.join(train_folder, song_type))
    song_name = os.path.join(train_folder, os.path.join(song_type, song_names[0]))
    y, sr = librosa.load(song_name)

    D = np.abs(librosa.stft(y))

    plot_spectrogram(D)

    RMScurve, ODF_RMS= rms_odf(D)
    ODF = sf_odf(D)

    plot_novoty_curve(RMScurve, ODF_RMS, ODF)

    #
    # RMScurve, ODF_RMS = rms_odf(x, fr, fs, Hop, h)
    # ODF = sf_odf(x, fr, fs, Hop, h)
