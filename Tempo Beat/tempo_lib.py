import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(folder = '../datasets/Ballroom/'):
    print('Loading Dataset')
    train_folder = os.path.join(folder, 'BallroomData')
    label_folder = os.path.join(folder, 'BallroomAnnotations')

    music_files = {}
    with open(os.path.join(train_folder, 'allBallroomFiles')) as f:
        for file_loc in f.readlines():
            _, genre, filename = file_loc.strip().split('/')
            if music_files.get(genre) == None:
                music_files[genre] = {filename: librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None)}
            else:
                music_files[genre][filename] = librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None)

    print('File Loaded!')
    for genre, music in music_files.items():
        print('\t', genre, len(music))

    return music_files
    # song_types = [item for item in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder,item))]


def get_fourier_tempogram(y, sr, tempo_second = 8, hop_length = 512, window_length = 2048):
    D = librosa.stft(y, hop_length=hop_length, win_length=window_length) # spectrogram
    onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=window_length, aggregate=np.median) # novetly curve

    tempo_frame = int(sr/hop_length) * tempo_second  # sub windows size with seconds
    tempo_hop = int(tempo_frame/8) # while librosa recommend to set as 1
    tempogram = librosa.stft(onset_env, hop_length=tempo_hop, win_length=tempo_frame, n_fft=1024)

    return tempogram, tempo_hop


def bpm_filter(tempogram, tempo_block_size, lower_bpm = 60, upper_bpm=200):
    lower_bound = int(lower_bpm / tempo_block_size)
    upper_bound = int(upper_bpm / tempo_block_size) + 1

    tempogram[:lower_bound, :] = 0
    tempogram[upper_bound:, :] = 0
    return tempogram

def top_tempo_select(order_tempo_freq, tempo_gap_rate=0.25):

    Top1_tempo = order_tempo_freq[0]

    Top1_tempo_plus, Top1_tempo_minus = Top1_tempo*(1+tempo_gap_rate), Top1_tempo*(1-tempo_gap_rate)
    filtered_ordered_freq = order_tempo_freq[(order_tempo_freq>= Top1_tempo_plus) | (order_tempo_freq <= Top1_tempo_minus)]

    Top2_tempo = filtered_ordered_freq[0]

    if Top1_tempo < Top2_tempo:
        Top1_tempo, Top2_tempo = Top2_tempo, Top1_tempo

    return Top1_tempo, Top2_tempo

def get_tempo(y, sr, tempo_gap_rate = 0.25,tempo_second = 8, hop_length = 512, window_length = 2048, tempogram_type='fourier'):

    if tempogram_type == 'fourier':
        tempogram, tempo_hop = get_fourier_tempogram(y, sr, tempo_second = tempo_second, hop_length = hop_length, window_length = window_length)

        tempo_freq = sr/hop_length/2 # tempogram's y-axis max value
        tempo_max_bpm = tempo_freq * 60 # tempogram's max bpm
        tempo_block_size = tempo_max_bpm / tempogram.shape[0] # get each value of tempogram's size

        filter_tempo_freq = bpm_filter(tempogram, tempo_block_size)  # only 60 ~ 200 bpm left
        filter_tempo_mean = np.mean(filter_tempo_freq, axis=1)  # get the average strength for each frequency
        ordered_freq = filter_tempo_mean.argsort()[::-1] * tempo_block_size  # in descending and in frequency unit

    elif tempogram_type == 'acf':
        acf_tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
        lag_block_size = 1/(sr/hop_length) # lag time for each y-axis
        acf_scale = np.array(range(acf_tempogram.shape[0])) * lag_block_size # all possible lag for this tempogram [box_size, box_size*2, box_size*3 ... box_size *400]
        acf_bpm = 60 / (acf_scale+1e-16) # convert lag to bpm

        possible_idx = (acf_bpm <= 200) & (acf_bpm >= 60) # index for bpm 60 ~ 200
        strength_order = np.mean(acf_tempogram, axis=1)[possible_idx].argsort()[::-1] # bpm tempogram order by strength in 60~200
        ordered_freq = acf_bpm[possible_idx][strength_order] # ordered bpm value in 60~200

    else:
        raise 'Tempogram Type Error!'


    return top_tempo_select(ordered_freq, tempo_gap_rate)


if __name__ == "__main__":

    music_files = load_dataset()

    for genre, music in music_files.items():
        for filename, (y, sr) in music.items():
            print(genre, filename)
            print(get_tempo(y, sr))
            print(get_tempo(y, sr, tempogram_type='acf'))
            break




