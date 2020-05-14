import test_n_gt_1, train_n_gt_1
import pickle
from sys import argv
import os
from hyper_params import *

def get_meta():
    if OnlySup:
        return language+paradigm + f'_only{Supervision}sup'
    meta = language+paradigm + f'_all-ns_{Supervision}sup'
    if minidict:
        meta += '_minidict'
    meta += f'_enhance_{enhance_iters}'
    meta += '_max'
    return meta

def get_covered_labels(data):
    if not include_only_covered_labels:
        return None
    covered_labels = set()
    for d in data:
        datum = d[0]
        datum = datum[datum.index('+')+1:]
        labels = [datum[:datum.index('+')], datum[datum.index('+') + 1:]]
        labels = [';'.join(l) for l in labels]
        assert len(labels)==2
        covered_labels.add(labels[0])
        covered_labels.add(labels[1])
    print(f'{len(covered_labels)} covered labels.')
    return covered_labels

def pack_params(train_module):
    return [train_module.model, train_module.enc_fwd_lstm, train_module.enc_bwd_lstm, train_module.dec_lstm,
            train_module.input_lookup, train_module.attention_w1,train_module.attention_w2, train_module.attention_v,
            train_module.decoder_w, train_module.decoder_b, train_module.output_lookup, train_module.VOCAB_SIZE]

def create_examples_per_label(added_examples):
    if not added_examples:
        return None
    added_examples_per_label = {}
    for relation in added_examples:
        label = relation.split()[0]
        if label not in added_examples_per_label:
            added_examples_per_label[label] = 0
        added_examples_per_label[label] += added_examples[relation]
    return added_examples_per_label

if __name__ == '__main__':
    meta = get_meta()
    assert language+paradigm in meta
    addition = argv[2] if len(argv)==3 else ''
    assert language in meta, paradigm in meta
    if OrigData:
        data_path = os.path.join(inflec_data_dir, f'{language}.um.{paradigm}.3.txt')
        model_path = os.path.join(model_dir_path, f'{language}.um.{paradigm}_model')
        write_path = os.path.join(model_dir_path, f'{language}.um.{paradigm}_output.txt')

    else:
        data_path = os.path.join(algo_outputs_path, f'data_{meta}.txt')
        model_path = os.path.join(model_dir_path, f'{meta}{addition}_model')
        write_path = os.path.join(model_dir_path, f'{meta}{addition}_output.txt')

    if OrigData:
        data, added_examples = train_n_gt_1.readdata(data_path)
    else:
        data, added_examples = train_n_gt_1.my_readdata(data_path)

    covered_labels = get_covered_labels(data)
    added_examples_per_label = create_examples_per_label(added_examples)
    train_n_gt_1.init()
    train_n_gt_1.train(train_n_gt_1.model, data)
    train_n_gt_1.model.save(model_path)
    pickle.dump((train_n_gt_1.int2char,train_n_gt_1.char2int,train_n_gt_1.VOCAB_SIZE),open("%s.obj.pkl" % model_path, "wb"))

    data, answers = test_n_gt_1.read_wrap(os.path.join(inflec_data_dir, f'{language}.um.{paradigm}.3.txt'),
                                          os.path.join(inflec_data_dir, f'{language}.um.{paradigm}.txt'))
    # int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % model_path, "rb"))
    int2char, char2int, VOCAB_SIZE = train_n_gt_1.int2char,train_n_gt_1.char2int,train_n_gt_1.VOCAB_SIZE
    list_of_stuff = pack_params(train_n_gt_1)
    test_n_gt_1.init_existing(list_of_stuff + [int2char, char2int])
    # test_n_gt_1.init()
    # test_n_gt_1.model.populate(model_path)
    all_labels = list(answers[0].keys())
    accuracy = test_n_gt_1.test(data, None, answers, write_path, added_examples, added_examples_per_label, covered_labels)
