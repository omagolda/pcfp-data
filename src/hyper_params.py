# HPs from main
algo_outputs_path = '../../morphodetection/initial_version/datasets'
model_dir_path = 'outputs'
inflec_data_dir = '../data'
language = 'ru'
paradigm = 'ADJ'
include_only_covered_labels = True
minidict = True
enhance_iters = 0
OrigData = False
OnlySup = False

# HPs from train
EPOCHS=50
BALANCE_NUM = 0

# HPs from test
WeightInVote = True
assert not OrigData or not WeightInVote     # both can't be true at once
