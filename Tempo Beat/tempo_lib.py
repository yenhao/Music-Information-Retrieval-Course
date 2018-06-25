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
                music_files[genre] = {filename: {'signal':librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None), 'label':0, 'beats_time':None, 'down_beats_time':None}}
            else:
                music_files[genre][filename] = {'signal':librosa.load(os.path.join(train_folder, file_loc.strip()), sr=None), 'label':0, 'beats_time':None, 'down_beats_time':None}

            #load labels
            with open(os.path.join(label_folder,os.path.join('ballroomGroundTruth', "".join(filename.split('.')[:-1])+'.bpm'))) as l:
                music_files[genre][filename]['label'] = int(l.read().strip())

            #load beats labels
            beat_label_folder = os.path.join(label_folder, os.path.join("BallroomAnnotations", "".join(filename.split('.')[:-1]) + ".beats"))
            with open(beat_label_folder) as f:
                lines = f.readlines()
                music_files[genre][filename]['beats_time'] = np.array([float(bt.strip().split()[0]) for bt in lines])
                music_files[genre][filename]['down_beats_time'] = np.array([float(bt.strip().split()[0]) for bt in lines if bt.strip().split()[1] == '1'])
                # music_files[genre][filename]['down_beats_time'] = []
                # for bt in f.readlines():
                #     print(bt.strip().split()[1])
                #     if bt.strip().split()[1] == '1':
                #         music_files[genre][filename]['down_beats_time'].append(float(bt.strip().split()[0]))


    print('File Loaded!')
    for genre, music in music_files.items():
        print('\t', genre, len(music))

    return music_files
    # song_types = [item for item in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder,item))]


def get_fourier_tempogram(y, sr, tempo_second = 8, hop_length = 512, window_length = 2048):
    D = librosa.stft(y, hop_length=hop_length, win_length=window_length) # spectrogram
    onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=window_length, aggregate=np.median)  # novetly curve   np.median/ np.mean/ None

    tempo_frame = int(sr/hop_length) * tempo_second  # sub windows size with seconds
    tempo_hop = int(tempo_frame/8) # while librosa recommend to set as 1
    tempogram = librosa.stft(onset_env, hop_length=tempo_hop, win_length=tempo_frame, n_fft=1024)

    return tempogram, tempo_hop


def bpm_filter(tempogram, tempo_block_size, lower_bpm = 80, upper_bpm=210):
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

        filter_tempo_freq = bpm_filter(tempogram, tempo_block_size)  # only 60 ~ 240 bpm left
        filter_tempo_mean = np.mean(filter_tempo_freq, axis=1)  # get the average strength for each frequency
        ordered_freq = filter_tempo_mean.argsort()[::-1] * tempo_block_size  # in descending and in frequency unit

    elif tempogram_type == 'acf':
        onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=window_length, aggregate=np.median)  # novetly curve   np.median/ np.mean/ None
        acf_tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
        # acf_tempogram = librosa.feature.tempogram(y=y, sr=sr, hop_length=hop_length)
        lag_block_size = 1/(sr/hop_length) # lag time for each y-axis
        acf_scale = np.array(range(acf_tempogram.shape[0])) * lag_block_size # all possible lag for this tempogram [box_size, box_size*2, box_size*3 ... box_size *400]
        acf_bpm = 60 / (acf_scale+1e-16) # convert lag to bpm

        possible_idx = (acf_bpm <= 210) & (acf_bpm >= 80) # index for bpm 60 ~ 240
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


def get_Precision(label, pred):
    tp, fp = 0, 0
    for p in pred:
        if label[(label >= p - 0.07) & (label <= p + 0.07)].size > 0:  # match
            tp += 1
        else:  # false alarm
            fp += 1
    return tp / (tp + fp) if tp > 0 else 0


def get_Recall(label, pred):
    tp, fn = 0, 0
    for l in label:
        if pred[(pred >= l - 0.07) & (pred <= l + 0.07)].size > 0:  # match, tp
            tp += 1
        else:  # miss, false negative
            fn += 1
    return tp / (tp + fn) if tp > 0 else 0

def get_F1_score(label, pred):
    precision = get_Precision(label, pred)
    recall = get_Recall(label, pred)
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

## for downbeats
def get_possible_frame(beats_strong_list, gap=4):
    skip = gap
    variation = gap

    frame_candidate = []
    for i in range(gap):  # different start point
        gap_bar = slide_window(beats_strong_list[i:], gap)
        num_1 = 0
        for bar in gap_bar:
            # count strongest in first
            bar = np.array(bar)
            if np.argsort(bar)[0] == gap - 1:  # biggest one is 1
                num_1 += 1  # count on 1
        frame_candidate.append(num_1 / len(gap_bar))  # average 1 in first placed
    return (gap, frame_candidate)


def slide_window(strong_list, skip):
    candidate = strong_list[:skip]
    c_len = len(candidate)
    if c_len < skip:  # to the end
        candidate += [0] * (skip - c_len)
        return [candidate]
    return [candidate] + slide_window(strong_list[skip:], skip)


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

            # beats tracking
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr) # global tempo and the frame of beat
            beats_time = librosa.frames_to_time(beats, sr=sr) # get the time of beat
            beats_time = beats_time[beats_time<=30] # limited on smaller than 30

            beats_F1_score = get_F1_score(sig_label['beats_time'], beats_time)
            precision = get_Precision(sig_label['beats_time'], beats_time)
            recall = get_Recall(sig_label['beats_time'], beats_time)
            print(precision)
            print(recall)
            print(beats_F1_score)


            # downbeats
            hop_length = 512
            onset_env = librosa.onset.onset_strength(y, sr=sr, hop_length=hop_length, n_fft=1024)
            onset_diff = np.append(np.zeros(1), np.diff(onset_env))  # get the difference of onset
            onset_diff[onset_diff < 0] = 0  # ReLU

            # take strong beats time
            onset_diff_time = []
            for i, strong in enumerate(onset_diff):
                if strong > 0:
                    onset_diff_time.append((i / (sr / hop_length), strong))

            # get strong degree of each beats time
            max_len = len(onset_diff_time)
            strong_weak_beats_list = []
            for bt in beats_time:
                for i in range(max_len):
                    time, strong = onset_diff_time[i]
                    if time <= bt:  # small than bt
                        if i + 1 < max_len:
                            if onset_diff_time[i + 1][0] > bt:  # next bigger than bt
                                # take value as strong value
                                strong_weak_beats_list.append(strong)
                                break  # search for next bt's strong value

            four_bar = get_possible_frame(strong_weak_beats_list, 4) # 4_bar's candidate
            three_bar = get_possible_frame(strong_weak_beats_list, 3) # 3_bar's candidate

            if max(four_bar[1]) > max(three_bar[1]):  # is 4
                down_beats_idx = (4, four_bar[1].index(max(four_bar[1])))
            else:  # is 3
                down_beats_idx = (3, three_bar[1].index(max(three_bar[1])))

            # down_beats_idx = skip, start_idx

            downbeats_sec = np.array([i for i in beats_time[down_beats_idx[1]::down_beats_idx[0]]])

            print(downbeats_sec)
            print(sig_label['down_beats_time'])
            downbeats_F1_score = get_F1_score(sig_label['down_beats_time'], downbeats_sec)
            downbeats_precision = get_Precision(sig_label['down_beats_time'], downbeats_sec)
            downbeats_recall = get_Recall(sig_label['down_beats_time'], downbeats_sec)
            print(downbeats_F1_score)
            print(downbeats_precision)
            print(downbeats_recall)

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
                                    'acf_saliency_p_score_mul4':[acf_p_score_mut_4[0]], 'acf_alotc_p_score_mul4':[acf_p_score_mut_4[1]],
                                    'beats_F1_score':beats_F1_score, 'beats_precision':precision, 'beats_recall':recall,
                                    'downbeats_F1_score': downbeats_F1_score, 'downbeats_precision': downbeats_precision, 'downbeats_recall': downbeats_recall}))

    p_score_res_pd.to_pickle('ballroom_res_pd_80_210_median.pkl')






