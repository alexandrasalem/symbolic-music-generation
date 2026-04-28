import pandas as pd
from fontTools.misc.cython import returns
from scipy.stats import wilcoxon
import os

model_results = os.listdir('result_analyses/')
model_results = [item for item in model_results if "simplified" not in item]
model_results = [item for item in model_results if "theme" not in item]
bass_gt = pd.read_csv("result_analyses/new_simplified_bass_files_c_midi_results.csv")
melody_gt = pd.read_csv("result_analyses/new_simplified_melody_files_c_midi_results.csv")

os.makedirs('result_wilcox/', exist_ok=True)

metrics = ['pitch_entropy', 'upc','unique_pitches', 'pr','pitch_interval','ctr','ctrp', 'note_count', 'ioi']
for model_result in model_results:
    mod = pd.read_csv("result_analyses/" + model_result)
    metric_stats = []
    metric_ps = []
    for metric in metrics:
        res_name = f'{metric}_sig_{model_result[:-4]}'
        if 'bass_results' in model_result or 'bass_samples_results' in model_result:
            gt = bass_gt[[metric]]
        elif 'melody_results' in model_result or 'melody_samples_results' in model_result:
            gt = melody_gt[[metric]]
        else:
            raise NotImplementedError
        res = mod[[metric]]
        my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
        metric_stats.append(my_stat.statistic[0])
        metric_ps.append(my_stat.pvalue[0])
    out_df = pd.DataFrame({'metric': metrics, 'metric_stats': metric_stats, 'metric_ps': metric_ps})
    out_df.to_csv(path_or_buf='result_wilcox/' + model_result + '_wilcox.csv')
    print("hi")

model_results = os.listdir('result_analyses/')
model_results = [item for item in model_results if "simplified" not in item]
model_results = [item for item in model_results if "theme" in item]
bass_gt = pd.read_csv("result_analyses/new_simplified_bass_files_c_midi_theme_results.csv")
melody_gt = pd.read_csv("result_analyses/new_simplified_melody_files_c_midi_theme_results.csv")

os.makedirs('result_theme_wilcox/', exist_ok=True)

metrics = ['pitch_entropy', 'upc','unique_pitches', 'pr','pitch_interval','ctr','ctrp', 'note_count', 'ioi']
for model_result in model_results:
    mod = pd.read_csv("result_analyses/" + model_result)
    metric_stats = []
    metric_ps = []
    for metric in metrics:
        res_name = f'{metric}_sig_{model_result[:-4]}'
        if 'bass' in model_result:
            gt = bass_gt[[metric]]
        elif 'melody' in model_result:
            gt = melody_gt[[metric]]
        else:
            raise NotImplementedError
        res = mod[[metric]]
        my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
        metric_stats.append(my_stat.statistic[0])
        metric_ps.append(my_stat.pvalue[0])
    out_df = pd.DataFrame({'metric': metrics, 'metric_stats': metric_stats, 'metric_ps': metric_ps})
    out_df.to_csv(path_or_buf='result_theme_wilcox/' + model_result + '_wilcox.csv')
    print("hi")


print("upc")
data = pd.read_csv('results_unique_pitch_ratio.csv')
comparisons = ['nochords_unique_pitch_ratio', 'chord2ind_unique_pitch_ratio', 'chord2sequentialbass_first_unique_pitch_ratio', 'chord2sequentialmelody_first_unique_pitch_ratio', 'chord2jointdecoder_unique_pitch_ratio', 'chord2symmetricdecoder_unique_pitch_ratio'] #["model_1_unique_pitch_ratio","model_2_unique_pitch_ratio","model_3a_unique_pitch_ratio","model_3b_unique_pitch_ratio"] #"model_4b_unique_pitch_ratio", "model_4c_unique_pitch_ratio"
gt = data[['gt_unique_pitch_ratio']]
for comparison in comparisons:
    res = data[[comparison]]
    my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
    print(comparison)
    print(my_stat)

print("\n\n")

print("upc theme")
data = pd.read_csv('results_theme_unique_pitch_ratio.csv')
comparisons = ['nochords_unique_pitch_ratio', 'chord2ind_unique_pitch_ratio', 'chord2sequentialbass_first_unique_pitch_ratio', 'chord2sequentialmelody_first_unique_pitch_ratio', 'chord2jointdecoder_unique_pitch_ratio', 'chord2symmetricdecoder_unique_pitch_ratio'] #["model_1_unique_pitch_ratio","model_2_unique_pitch_ratio","model_3a_unique_pitch_ratio","model_3b_unique_pitch_ratio"] #"model_4b_unique_pitch_ratio", "model_4c_unique_pitch_ratio"
gt = data[['gt_unique_pitch_ratio']]
for comparison in comparisons:
    res = data[[comparison]]
    my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
    print(comparison)
    print(my_stat)

print("\n\n")

print("pcs")
data = pd.read_csv('results_pcs.csv')
comparisons = ['nochords_pcs', 'chord2ind_pcs', 'chord2sequentialbass_first_pcs', 'chord2sequentialmelody_first_pcs', 'chord2jointdecoder_pcs', 'chord2symmetricdecoder_pcs', 'gt_pcs']
gt = data[['gt_pcs']]
for comparison in comparisons:
    res = data[[comparison]]
    my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
    print(comparison)
    print(my_stat)

print("\n\n")

print("pcs theme")
data = pd.read_csv('results_theme_pcs.csv')
comparisons = ['nochords_pcs', 'chord2ind_pcs', 'chord2sequentialbass_first_pcs', 'chord2sequentialmelody_first_pcs', 'chord2jointdecoder_pcs', 'chord2symmetricdecoder_pcs', 'gt_pcs']
gt = data[['gt_pcs']]
for comparison in comparisons:
    res = data[[comparison]]
    my_stat = wilcoxon(gt, res, alternative='two-sided', nan_policy='omit')
    print(comparison)
    print(my_stat)



