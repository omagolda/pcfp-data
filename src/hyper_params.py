# HPs from main
algo_outputs_path = '../../morphodetection/initial_version/outputs'
exp_dir = ''
model_dir_path = 'outputs'
inflec_data_dir = '../data'
language = 'ru'
paradigm = 'ADJ'
include_only_covered_labels = False
minidict = True
OrigData = False    # set True to use the original pcfp-data
scoring_threshold = 0
# enhance_iters = 0
# OnlySup = False
# Supervision = 5

# HPs from train
EPOCHS=50
BALANCE_NUM = 0

# HPs from test
WeightInVote = True
assert not OrigData or not WeightInVote     # both can't be true at once
