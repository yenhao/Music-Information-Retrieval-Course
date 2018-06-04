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

def plot_novoty_curve(RMScurve=None, ODF_RMS=None, ODF=None, hop=512, sr=22050):

    if RMScurve.all() != None:
        plt.plot(range(RMScurve.shape[0]), RMScurve, label='RMScurve')
    if ODF_RMS.all() != None:
        plt.plot(range(RMScurve.shape[0]), ODF_RMS, label='ODF_RMS')
    if ODF.all() != None:
        plt.plot(range(RMScurve.shape[0]), ODF, label='ODF')

    plt.xlabel("Time")
    plt.ylabel("Onset")
    plt.legend()

    # time xtick

    x_len = RMScurve.shape[0]
    x_time = int(x_len / (sr/hop))
    # print(x_time)
    # print(int(x_time/5))
    # print([str(t) for t in range(0, x_time, 5)])
    plt.xticks(np.linspace(0, x_len, int(x_time/5)+1), [str(t) for t in range(0, x_time, 5)])

    plt.show()
    plt.close()

def plot_spectrogram(D):
    librosa.display.specshow(librosa.amplitude_to_db(D, ref = np.max), y_axis = 'log', x_axis = 'time')
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

def fourier_tempogram(y, sr=22050, hop=512, second = 8):
    D = librosa.stft(y)
    ODF = sf_odf(D)

    x_len = ODF.shape[0]
    print(x_len, sr/hop)
    x_fs= int(sr/hop) # about 43
    print(x_fs)
    print(x_fs*second)
    tempo = librosa.stft(ODF, n_fft=x_fs*second)

    return tempo



if __name__ == "__main__":

    song_type = song_types[0]
    song_names = os.listdir(os.path.join(train_folder, song_type))
    song_name = os.path.join(train_folder, os.path.join(song_type, song_names[0]))
    print(song_name)

    y, sr = librosa.load(song_name) # fs = 22050, hop = 512, window = 2048
    hop = 512
    window = 2048
    # D = np.abs(librosa.stft(y))
    D = librosa.stft(y)
    print(D.shape)
    # plot_spectrogram(D)

    RMScurve, ODF_RMS= rms_odf(np.abs(D))
    ODF = sf_odf(D)
    print(ODF.shape)
    # plot_novoty_curve(RMScurve, ODF_RMS, ODF)

    tempogram = fourier_tempogram(y, second=8)

    # print(tempogram)
    print(tempogram[:,0].shape)
    print(tempogram.shape)
    librosa.display.specshow(tempogram, sr=sr, hop_length=hop, x_axis='time', y_axis='tempo')
    plt.colorbar()
    plt.show()
    # print(tempogram[:,0])
    print(int(sr/hop)*8)
    acf_tempogram = librosa.istft(np.sqrt(np.abs(tempogram)), win_length=int(sr/hop)*8)
    print(acf_tempogram.shape)

    # librosa.display.specshow(acf_tempogram, sr=sr, hop_length=int(sr/hop)*8/4, x_axis='time', y_axis='lag')
    # plt.colorbar()
    # plt.show()

    ### Librosa function
    # o_env = librosa.onset.onset_strength(y=y, sr=sr)
    #
    # times = librosa.frames_to_time(np.arange(len(o_env)), sr=sr)
    # onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)

    # plt.figure(figsize=(20,20))
    # ax1 = plt.subplot(2, 1, 1)
    #
    # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis = 'time', y_axis = 'log')
    # plt.title('Power spectrogram')
    # plt.subplot(2, 1, 2, sharex=ax1)
    # plt.plot(times, o_env, label='Onset strength')
    # plt.vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle = '--', label = 'Onsets')
    # plt.axis('tight')
    # plt.legend(frameon=True, framealpha=0.75)
    # plt.show()

    # hop_length = 512
    # tempogram = librosa.feature.tempogram(onset_envelope=o_env, sr=sr, hop_length = hop_length)
    #
    # print(tempogram)
    # print(tempogram.shape)
    # Compute global onset autocorrelation
    # ac_global = librosa.autocorrelate(o_env, max_size=tempogram.shape[0])
    # ac_global = librosa.util.normalize(ac_global)
    # # Estimate the global tempo for display purposes
    # tempo = librosa.beat.tempo(onset_envelope=o_env, sr=sr,
    #                            hop_length=hop_length)[0]
    #
    # plt.figure(figsize=(8, 8))
    # plt.subplot(4, 1, 1)
    # plt.plot(o_env, label='Onset strength')
    # plt.xticks([])
    # plt.legend(frameon=True)
    # plt.axis('tight')
    # plt.subplot(4, 1, 2)
    # # We'll truncate the display to a narrower range of tempi
    # librosa.display.specshow(tempogram, sr=sr, hop_length=hop_length,
    #                          x_axis='time', y_axis='tempo')
    # plt.axhline(tempo, color='w', linestyle='--', alpha=1,
    #             label='Estimated tempo={:g}'.format(tempo))
    # plt.legend(frameon=True, framealpha=0.75)
    # plt.subplot(4, 1, 3)
    # x = np.linspace(0, tempogram.shape[0] * float(hop_length) / sr,
    #                 num=tempogram.shape[0])
    # plt.plot(x, np.mean(tempogram, axis=1), label='Mean local autocorrelation')
    # plt.plot(x, ac_global, '--', alpha=0.75, label='Global autocorrelation')
    # plt.xlabel('Lag (seconds)')
    # plt.axis('tight')
    # plt.legend(frameon=True)
    # plt.subplot(4, 1, 4)
    # # We can also plot on a BPM axis
    # freqs = librosa.tempo_frequencies(tempogram.shape[0], hop_length=hop_length, sr=sr)
    # plt.semilogx(freqs[1:], np.mean(tempogram[1:], axis=1),
    #              label='Mean local autocorrelation', basex=2)
    # plt.semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
    #              label='Global autocorrelation', basex=2)
    # plt.axvline(tempo, color='black', linestyle='--', alpha=.8,
    #             label='Estimated tempo={:g}'.format(tempo))
    # plt.legend(frameon=True)
    # plt.xlabel('BPM')
    # plt.axis('tight')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()