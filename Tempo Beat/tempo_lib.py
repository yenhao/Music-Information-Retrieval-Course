import os, pickle
import pandas as pd
import numpy as np
import librosa
import librosa.display
import numpy as np


def load_dataset(folder = '../datasets/Ballroom/'):
    print('Loading Dataset')
    train_folder = os.path.join(folder, 'BallroomData')
    label_folder = os.path.join(folder, 'BallroomAnnotations')

    music_files = {}
    with open(os.path.join(train_folder, 'allBallroomFiles')) as f:
        for file_loc in f.readlines():
            _, genre, filename = file_loc.strip().split('/')
            genre = genre.split('-')[0]
            if music_files.get(genre) == None:
                music_files[genre] = {filename: {'signal':librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None), 'label':0}}
            else:
                music_files[genre][filename] = {'signal':librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None), 'label':0}

            #load labels
            with open(os.path.join(label_folder,os.path.join('ballroomGroundTruth', "".join(filename.split('.')[:-1])+'.bpm'))) as l:
                music_files[genre][filename]['label'] = int(l.read().strip())

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


def bpm_filter(tempogram, tempo_block_size, lower_bpm = 60, upper_bpm=240):
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

    if Top1_tempo > Top2_tempo:
        Top1_tempo, Top2_tempo = Top2_tempo, Top1_tempo

    return Top1_tempo, Top2_tempo # T1, T2

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
        onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=window_length,
                                                 aggregate=np.median)  # novetly curve
        acf_tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        # acf_tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
        lag_block_size = 1/(sr/hop_length) # lag time for each y-axis
        acf_scale = np.array(range(acf_tempogram.shape[0])) * lag_block_size # all possible lag for this tempogram [box_size, box_size*2, box_size*3 ... box_size *400]
        acf_bpm = 60 / (acf_scale+1e-16) # convert lag to bpm

        possible_idx = (acf_bpm <= 240) & (acf_bpm >= 60) # index for bpm 60 ~ 200
        strength_order = np.mean(acf_tempogram, axis=1)[possible_idx].argsort()[::-1] # bpm tempogram order by strength in 60~200
        ordered_freq = acf_bpm[possible_idx][strength_order] # ordered bpm value in 60~200

    else:
        raise 'Tempogram Type Error!'


    return top_tempo_select(ordered_freq, tempo_gap_rate)

def p_score(t1,t2, y):
    saliency = t1 / (t1+t2)

    Tt1 = 1 if abs(y-t1)/y <= 0.08 else 0
    Tt2 = 1 if abs(y-t2)/y <= 0.08 else 0

    p = saliency * Tt1 + (1 - saliency)*Tt2

    # at least one tempo correct
    alotc = 1 if abs(y-t1)/y < 0.08 else 0

    if alotc == 0:
        alotc = 1 if abs(y-t2)/y < 0.08 else 0

    return p, alotc, t2/t1, t1/y, t2/y

if __name__ == "__main__":

    music_files = load_dataset()

    p_score_res_pd = pd.DataFrame()

    for genre, music in music_files.items():
        for filename, sig_label in music.items():
            print(genre, filename)
            (y, sr) = sig_label['signal']
            label = sig_label['label']
            print(label)

            # Fourier
            fourier_tempo = get_tempo(y, sr)
            fourier_p_score = p_score(fourier_tempo[0], fourier_tempo[1], label)
            fourier_p_score_div_2 = p_score(fourier_tempo[0]/2, fourier_tempo[1]/2, label)
            fourier_p_score_div_3 = p_score(fourier_tempo[0]/3, fourier_tempo[1]/3, label)
            fourier_p_score_div_4 = p_score(fourier_tempo[0]/4, fourier_tempo[1]/4, label)
            fourier_p_score_mut_2 = p_score(2 * fourier_tempo[0], 2 * fourier_tempo[1], label)
            fourier_p_score_mut_3 = p_score(3 * fourier_tempo[0], 3 * fourier_tempo[1], label)
            fourier_p_score_mut_4 = p_score(4 * fourier_tempo[0], 4 * fourier_tempo[1], label)
            print(fourier_tempo)
            print(fourier_p_score)

            # ACF
            acf_tempo = get_tempo(y, sr, tempogram_type='acf')
            acf_p_score = p_score(acf_tempo[0], acf_tempo[1], label)
            acf_p_score_div2 = p_score(acf_tempo[0]/2, acf_tempo[1]/2, label)
            acf_p_score_div3 = p_score(acf_tempo[0]/3, acf_tempo[1]/3, label)
            acf_p_score_div4 = p_score(acf_tempo[0] / 4, acf_tempo[1] / 4, label)
            acf_p_score_mut_2 = p_score(2 * acf_tempo[0], 2 * acf_tempo[1], label)
            acf_p_score_mut_3 = p_score(3 * acf_tempo[0], 3 * acf_tempo[1], label)
            acf_p_score_mut_4 = p_score(4 * acf_tempo[0], 4 * acf_tempo[1], label)
            print(acf_tempo)
            print(acf_p_score)



            # save to pandas
            p_score_res_pd = p_score_res_pd.append(pd.DataFrame({'genre':[genre], 'music':[filename], 'label' :[label],
                                    'fourier_tempo':[fourier_tempo], 'acf_tempo':[acf_tempo],
                                    'fourier_saliency_p_score':[fourier_p_score[0]], 'fourier_alotc_p_score':[fourier_p_score[1]],
                                    'acf_saliency_p_score':[acf_p_score[0]], 'acf_alotc_p_score':[acf_p_score[1]],
                                    'fourier_t21_rate':[fourier_p_score[2]], 'acf_t21_rate':[acf_p_score[2]],
                                    'fourier_t1g_rate':[fourier_p_score[3]], 'acf_t1g_rate':[acf_p_score[3]],
                                    'fourier_t2g_rate':[fourier_p_score[4]], 'acf_t2g_rate':[acf_p_score[4]],
                                    'fourier_saliency_p_score_div2':[fourier_p_score_div_2[0]], 'fourier_alotc_p_score_div2':[fourier_p_score_div_2[1]],
                                    'acf_saliency_p_score_div2':[acf_p_score_div2[0]], 'acf_alotc_p_score_div2':[acf_p_score_div2[1]],
                                    'fourier_saliency_p_score_div3':[fourier_p_score_div_3[0]], 'fourier_alotc_p_score_div3':[fourier_p_score_div_3[1]],
                                    'acf_saliency_p_score_div3':[acf_p_score_div3[0]], 'acf_alotc_p_score_div3':[acf_p_score_div3[1]],
                                    'fourier_saliency_p_score_div4':[fourier_p_score_div_4[0]], 'fourier_alotc_p_score_div4':[fourier_p_score_div_4[1]],
                                    'acf_saliency_p_score_div4':[acf_p_score_div4[0]], 'acf_alotc_p_score_div4':[acf_p_score_div4[1]],
                                    'fourier_saliency_p_score_mul2':[fourier_p_score_mut_2[0]], 'fourier_alotc_p_score_mul2':[fourier_p_score_mut_2[1]],
                                    'acf_saliency_p_score_mul2':[acf_p_score_mut_2[0]], 'acf_alotc_p_score_mul2':[acf_p_score_mut_2[1]],
                                    'fourier_saliency_p_score_mul3':[fourier_p_score_mut_3[0]], 'fourier_alotc_p_score_mul3':[fourier_p_score_mut_3[1]],
                                    'acf_saliency_p_score_mul3':[acf_p_score_mut_3[0]], 'acf_alotc_p_score_mul3':[acf_p_score_mut_3[1]],
                                    'fourier_saliency_p_score_mul4':[fourier_p_score_mut_4[0]], 'fourier_alotc_p_score_mul4':[fourier_p_score_mut_4[1]],
                                    'acf_saliency_p_score_mul4':[acf_p_score_mut_4[0]], 'acf_alotc_p_score_mul4':[acf_p_score_mut_4[1]]}))

    p_score_res_pd.to_pickle('ballroom_res_pd.pkl')






