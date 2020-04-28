import test_n_gt_1, train_n_gt_1
import pickle
from sys import argv
import os

algo_outputs_path = '../../morphodetection/initial_version/datasets'
model_dir_path = 'outputs'
inflec_data_dir = '../data'
language = 'ru'
paradigm = 'N'
include_only_covered_labels = True

def get_covered_labels():
    if not include_only_covered_labels:
        return None
    if language+paradigm == 'esV':
        covered_labels = {'V;COND;3;SG', 'V;SBJV;PRS;2;SG', 'V.PTCP;PST;MASC;PL', 'V.PTCP;PST;FEM;PL', 'V;IND;PRS;1;PL',
                          'V;POS;IMP;1;PL', 'V;POS;IMP;3;SG', 'V;IND;PRS;1;SG', 'V;IND;PST;3;SG;IPFV', 'V;IND;PST;3;PL;IPFV',
                          'V;SBJV;PST;1;SG;LGSPEC1', 'V;IND;PST;3;PL;PFV', 'V;IND;FUT;3;PL', 'V;IND;PST;3;SG;PFV', 'V;NFIN',
                          'V;IND;FUT;3;SG', 'V;SBJV;PRS;3;PL', 'V;SBJV;PST;3;SG', 'V;COND;3;PL', 'V;POS;IMP;2;SG', 'V;POS;IMP;3;PL',
                          'V;IND;PST;1;SG;IPFV', 'V;SBJV;PRS;1;SG', 'V;COND;1;SG', 'V;IND;PRS;3;PL', 'V.PTCP;PST;MASC;SG',
                          'V.PTCP;PST;FEM;SG', 'V.CVB;PRS', 'V;SBJV;PST;3;PL;LGSPEC1', 'V;IND;PRS;2;SG', 'V;SBJV;PST;1;SG',
                          'V;SBJV;PRS;3;SG', 'V;IND;PRS;3;SG', 'V;SBJV;PST;3;SG;LGSPEC1', 'V;SBJV;PRS;1;PL'}
    elif language+paradigm == 'fiN':
        covered_labels = {'N;AT+ESS;SG', 'N;NOM;SG', 'N;PRT;PL', 'N;IN+ABL;SG', 'N;IN+ALL;PL', 'N;AT+ALL;SG', 'N;IN+ABL;PL',
                          'N;IN+ESS;PL', 'N;ACC;PL', 'N;GEN;SG', 'N;NOM;PL', 'N;AT+ESS;PL', 'N;IN+ALL;SG', 'N;FRML;SG', 'N;ACC;SG',
                          'N;GEN;PL', 'N;IN+ESS;SG', 'N;PRT;SG'}
    elif language+paradigm == 'deN' or language+paradigm == 'ruN':
        covered_labels = None
    elif language+paradigm == 'frV':
        covered_labels = {'V;COND;3;SG', 'V;IND;PST;3;PL;IPFV', 'V;IND;PRS;2;PL', 'V;IND;PRS;1;PL', 'V;IND;PST;3;SG;IPFV',
                          'V;POS;IMP;1;PL', 'V;IND;PRS;3;PL', 'V;NFIN', 'V;POS;IMP;2;PL', 'V;SBJV;PRS;3;PL', 'V;POS;IMP;2;SG',
                          'V;IND;PST;3;PL;PFV', 'V;IND;PST;3;SG;PFV', 'V;IND;PRS;2;SG', 'V;SBJV;PRS;1;SG', 'V;IND;FUT;3;PL',
                          'V;IND;FUT;3;SG', 'V.PTCP;PRS', 'V.PTCP;PST', 'V;IND;PRS;1;SG', 'V;IND;PRS;3;SG', 'V;SBJV;PRS;3;SG'}
    elif language+paradigm == 'ruV':
        # covered_labels = {'V;PRS;3;PL', 'V;PST;SG;NEUT', 'V;PST;PL', 'V.CVB;PST', 'V;PST;SG;MASC', 'V;PST;SG;FEM',
        #                   'V;PRS;1;PL', 'V;PRS;3;SG', 'V;PRS;1;SG', 'V.CVB;PRS', 'V;NFIN'}
        # down is for the separate FUT and PRS case
        covered_labels = {'V;NFIN', 'V;PRS;3;PL', 'V;PST;PL', 'V;FUT;1;PL', 'V;PRS;1;SG', 'V;IMP;2;SG', 'V;PRS;3;SG',
                          'V;PST;SG;MASC', 'V;PST;SG;NEUT', 'V;FUT;3;SG', 'V.CVB;PRS', 'V.CVB;PST', 'V;PST;SG;FEM'}
    elif language+paradigm == 'fiV':
        covered_labels = {'V;PASS;PST;POS;IND', 'V.PTCP;PASS;PRS', 'V;NFIN', 'V;ACT;PRS;POS;IND;3;PL', 'V.PTCP;ACT;PRS',
                          'V;ACT;PST;POS;IND;1;SG', 'V;ACT;PST;POS;IND;3;SG', 'V;ACT;PST;POS;IND;3;PL', 'V;ACT;PRS;POS;IND;2;SG',
                          'V.PTCP;PASS;PST', 'V;ACT;PRS;POS;COND;1;SG', 'V;PASS;PRS;POS;IND', 'V;ACT;PRS;POS;IMP;2;SG',
                          'V;ACT;PRS;POS;COND;3;PL', 'V;ACT;PRS;POS;COND;3;SG', 'V;PASS;PRS;POS;COND', 'V;ACT;PRS;POS;IND;1;SG',
                          'V.PTCP;ACT;PST', 'V;ACT;PRS;POS;IND;3;SG'}
    else:
        raise NotImplementedError
    return covered_labels

def pack_params(train_module):
    return [train_module.model, train_module.enc_fwd_lstm, train_module.enc_bwd_lstm, train_module.dec_lstm,
            train_module.input_lookup, train_module.attention_w1,train_module.attention_w2, train_module.attention_v,
            train_module.decoder_w, train_module.decoder_b, train_module.output_lookup, train_module.VOCAB_SIZE]

def create_examples_per_label(added_examples):
    added_examples_per_label = {}
    for relation in added_examples:
        label = relation.split()[0]
        if label not in added_examples_per_label:
            added_examples_per_label[label] = 0
        added_examples_per_label[label] += added_examples[relation]
    return added_examples_per_label

if __name__ == '__main__':
    meta = argv[1]
    assert language+paradigm in meta
    addition = argv[2] if len(argv)==3 else ''
    assert language in meta, paradigm in meta
    data_path = os.path.join(algo_outputs_path, f'data_{meta}.txt')
    model_path = os.path.join(model_dir_path, f'{meta}{addition}_model')
    write_path = os.path.join(model_dir_path, f'{meta}{addition}_output.txt')

    data, added_examples = train_n_gt_1.my_readdata(data_path)
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
    accuracy = test_n_gt_1.test(data, None, answers, write_path, added_examples, added_examples_per_label, get_covered_labels())




