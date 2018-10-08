import os
import os.path
from math import *
import numpy as np
from numpy import genfromtxt

# Load the result file in the following format
#           lang0     lang1     lang2     lang3     lang4     lang5     lang6     lang7     lang8     lang9
# <utt-id>  <score0>  <score1>  <score2>  <score3>  <score4>  <score5>  <score6>  <score7>  <score8>  <score9>

# The language identity is defined as:
# {'ct-cn':'3', 'id-id':'4', 'ja-jp':'5', 'ko-kr':'6', 'ru-ru':'7', 'vi-vn':'8', 'zh-cn':'9', 'Kazak':'0', 'Tibet':'1', 'Uyghu':'2'}

langnum = 10
dictl = {'0': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10}

def genererate_test_score(path_output_file, path_reference_file,score_matrix):

    scores=score_matrix

    with open(path_output_file, 'w') as f:
        for ref_row, score_row in zip(get_table(path_reference_file), scores):
            score_str = np.array2string(score_row)[1:-1].split()
            ch = ref_row[0].decode() + "\t" + "\t".join(score_str) + "\n"
            f.write(ch)

def get_table(file):
    from numpy import genfromtxt
    with open(file, 'r') as f:
        header = f.readline()
    column_names = header.split()
    array = genfromtxt(file, dtype=None, skip_header=1, names=column_names)
    return array

def lid_format_utt(file):
    lang_dict = {'ct-cn': 'lang3', 'id-id': 'lang4', 'ja-jp': 'lang5', 'ko-kr': 'lang6', 'ru-ru': 'lang7',
                 'vi-vn': 'lang8', 'zh-cn': 'lang9', 'Kazak': 'lang0', 'Tibet': 'lang1', 'Uyghu': 'lang2'}

    file_path = file
    new_path = file+'_1'
    new_score = open(new_path, 'w')
    new_score.write('      lang0    lang1    lang2    lang3    lang4    lang5    lang6    lang7    lang8    lang9 \n')

    with open(file_path, 'r') as lines:
        for col in [line.strip().split() for line in lines]:
            line_id = col[0]
            lang_id = lang_dict[line_id[0:5]]
            new_score.write(lang_id + ' ' + ' '.join(col[1:12]) + '\n')

    new_score.close()

# Load scoring file and label.scp.
def Loaddata(fin, langnum):
    x = []
    for i in range(langnum + 1):
        x.append(0)

    fin = open(fin, 'r')
    lines = fin.readlines()
    fin.close()

    data = []
    for line in lines[1:]:
        part = line.split()
        x[0] = part[0].split('g')[1].split('_')[0]
        for i in range(langnum):
            x[i + 1] = part[i + 1]
        data.append(x)
        x = []
        for i in range(langnum + 1):
            x.append(0)

    datas = []
    for ll in data:
        for lb in range(langnum):
            datas.append([dictl[ll[0][0]], lb + 1, float(ll[lb + 1])])

    # score normalized to [0, 1]
    for i in range(int(len(datas) / langnum)):
        sum = 0
        for j in range(langnum):
            k = i * langnum + j
            sum += exp(datas[k][2])
        for j in range(langnum):
            k = i * langnum + j
            datas[k][2] = exp(datas[k][2]) / sum

    return datas

# Compute Cavg.
# data: matrix for result scores, assumed within [0,1].
# sn: number of bins in Cavg calculation.
def CountCavg(data, sn=20, lgn=4):
    Cavg = [0.0] * (sn + 1)
    # Cavg: Every element is the Cavg of the corresponding precision
    precision = 1.0 / sn
    for section in range(sn + 1):
        threshold = section * precision
        target_Cavg = [0.0] * lgn
        # target_Cavg: P_Target * P_Miss + sum(P_NonTarget*P_FA)

        for language in range(lgn):
            P_FA = [0.0] * lgn
            P_Miss = 0.0
            # compute P_FA and P_Miss
            LTm = 0.0;
            LTs = 0.0;
            LNm = 0.0;
            LNs = [0.0] * lgn;
            for line in data:
                language_label = language + 1
                if line[0] == language_label:
                    if line[1] == language_label:
                        LTm += 1
                        if line[2] < threshold:
                            LTs += 1
                    for t in range(lgn):
                        if not t == language:
                            if line[1] == t + 1:
                                if line[2] > threshold:
                                    LNs[t] += 1
            LNm = LTm
            for Ln in range(lgn):
                P_FA[Ln] = LNs[Ln] / LNm

            P_Miss = LTs / LTm
            P_NonTarget = 0.5 / (lgn - 1)
            P_Target = 0.5
            target_Cavg[language] = P_Target * P_Miss + P_NonTarget * sum(P_FA)

        for language in range(lgn):
            Cavg[section] += target_Cavg[language] / lgn

    return Cavg, min(Cavg)

###################
def Compute_Cavg(file_path,ref_file):
    score_matrix=genfromtxt(file_path, delimiter=',')
    score_matrix=np.log(score_matrix+0.0000000000000000000000000001)


    if not os.path.isdir(r'lid_score'):
            os.makedirs(r'lid_score')

    path_output_file= './lid_score/temp'
    path_reference_file="./Lists/Reference/"+ref_file+'_list.txt'


    genererate_test_score(path_output_file, path_reference_file,score_matrix)
    lid_format_utt(path_output_file)

    data = Loaddata(path_output_file+'_1', langnum)
    # default precision as 20 bins, langnum languages
    cavg, mincavg = CountCavg(data, 20, langnum)

    print("Minimal Cavg is: " + str(round(mincavg, 4)) + '\n' )

    return str(round(mincavg, 4))

