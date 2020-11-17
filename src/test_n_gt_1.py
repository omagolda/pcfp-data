from sys import stdout, argv
from collections import Counter

import dynet as dy
import random
import pickle
from hyper_params import *

EOS = "<EOS>"

int2char = [EOS,'+']
char2int = {EOS:0,'+':1}

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100

def init():
    global model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, attention_w1,\
    attention_w2,attention_v,decoder_w,decoder_b,output_lookup, VOCAB_SIZE
    VOCAB_SIZE = len(char2int)

    model = dy.Model()

    enc_fwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE,
                                  model)
    enc_bwd_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE,
                                  model)

    dec_lstm = dy.LSTMBuilder(LSTM_NUM_OF_LAYERS, STATE_SIZE*2+EMBEDDINGS_SIZE,
                              STATE_SIZE, model)

    input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
    attention_w1 = model.add_parameters( (ATTENTION_SIZE, STATE_SIZE*2))
    attention_w2 = model.add_parameters( (ATTENTION_SIZE,
                                          STATE_SIZE*LSTM_NUM_OF_LAYERS*2))
    attention_v = model.add_parameters( (1, ATTENTION_SIZE))
    decoder_w = model.add_parameters( (VOCAB_SIZE, STATE_SIZE))
    decoder_b = model.add_parameters( (VOCAB_SIZE))
    output_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))

def init_existing(list_of_stuff):
    global model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, attention_w1,\
    attention_w2,attention_v,decoder_w,decoder_b,output_lookup, VOCAB_SIZE, int2char, char2int

    model, enc_fwd_lstm, enc_bwd_lstm, dec_lstm, input_lookup, attention_w1,\
    attention_w2,attention_v,decoder_w,decoder_b,output_lookup, VOCAB_SIZE, int2char, char2int = list_of_stuff


def embed_sentence(sentence):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]

    global input_lookup

    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state

    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))

    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]

    return vectors


def attend(input_mat, state, w1dt):
    global attention_w2
    global attention_v
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2*dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


def decode(dec_lstm, vectors, output):
    output = [EOS] + list(output) + [EOS]
    output = [char2int[c] for c in output]

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(vectors)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE*2), last_output_embeddings]))
    loss = []

    for char in output:
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector)
        last_output_embeddings = output_lookup[char]
        loss.append(-dy.log(dy.pick(probs, char)))
    loss = dy.esum(loss)
    return loss


def generate(in_seq, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    embedded = embed_sentence(in_seq)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings]))

    out = ''
    count_EOS = 0
    for i in range(len(in_seq)*2):
        if count_EOS == 2: break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate([attend(input_mat, s, w1dt), last_output_embeddings])
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def get_loss(input_sentence, output_sentence, enc_fwd_lstm, enc_bwd_lstm, dec_lstm):
    dy.renew_cg()
    embedded = embed_sentence(input_sentence)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)
    return decode(dec_lstm, encoded, output_sentence)


def train(model, data):
    trainer = dy.SimpleSGDTrainer(model)
    for n in range(5):
        totalloss = 0
        random.shuffle(data)
        for i, io in enumerate(data):
            if i > 5000:
                break
            stdout.write('EPOCH %u: ex %u of %u\r' % (n+1,i+1,len(data)))
            input,output = io
            loss = get_loss(input, output, enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            totalloss += loss.value()
            loss.backward()
            trainer.update()
        print()
        print(totalloss/len(data))
        for input, output in data[:10]:
            print('input:',''.join(input),
                  'sys:',generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm),
                  'gold:',''.join(output))

def readtestdata(fn):
    data = [{}]
    labels = set()
    for line in open(fn, encoding='utf8'):
        line = line.strip('\n')
        if line == '':
            data.append({})
        else:
            wf, label = line.split('\t')
            labels.add(label)
            if wf == '':
                continue
            data[-1][label] = wf
    # data = [d for d in data if d != {}]
    # for d in data:
    #     for l in labels:
    #         if not l in d:
    #             d[l] = None
    # return [d for d in data if d != {}]

    return data

def lemma_readtestdata(fn):
    if fn.endswith('.lemma.txt'):
        data = []
        for line in open(fn, encoding='utf8'):
            if not line or len(line.split())>1:
                continue
            data.append({'V,NFIN': line.strip()})
    else:
        data = []
        former_lemma = None
        for line in open(fn, encoding='utf8'):
            line = line.strip('\n')
            if line == '':
                continue
            lemma, wf, label = line.split('\t')
            if language == 'ru':
                label = label.replace('FUT', 'PRS')
                if label.startswith('V.PTCP'):
                    continue
            if language == 'tr' and 'NEG' in label:
                continue
            if len(lemma.split())>1:
                continue
            if lemma != former_lemma:
                data.append({})
            former_lemma = lemma
            if len(wf.split()) != 1:
                continue
            if wf == '':
                raise NotImplementedError
            data[-1][label] = wf
    return data


def vote(outputs):
    outputs = [output for output in outputs if output]
    return Counter(outputs).most_common()[0][0]

def weight_vote(outputs, relation_distrusts, labels_distrust):
    # assert len(outputs)==len(distrusts)
    count = Counter(outputs)
    _ = count.pop(None, None)

    if count.most_common()[0][1] != 1:
        return Counter(outputs).most_common()[0][0]

    elif set(relation_distrusts) != {None}:
        rds = [d if d is not None else 10000000 for d in relation_distrusts]
        return outputs[relation_distrusts.index(min(rds))]

    elif set(labels_distrust) != {None}:
        lds = [d if d is not None else 10000000 for d in labels_distrust]
        return outputs[labels_distrust.index(min(lds))]

    else:
        return random.choice(outputs)

def test(partial_data, answers, write_path, relations_distrust, labels_distrust):
    with open(write_path, 'w', encoding='utf8') as f:
        predicts = []
        for i, d in enumerate(partial_data):
            p = {}
            dy.renew_cg()
            forms = [[c for c in wf] + ['+'] + l.split(',')
                     for l,wf in d.items() if wf != None]
            labels = [';'.join(l.split(',')) for l,wf in d.items() if wf != None]

            all_relevant_labels = answers[i].keys()
            for l in all_relevant_labels:
                # if d[l] == None:
                if d.get(l, None) == None:

                    templ = l.replace(';', ',')
                    inputs = [f + ['+'] + templ.split(',') for f in forms]
                    relations = []
                    outputs = []
                    for i, input in enumerate(inputs):
                        relations.append(labels[i] + ' ' + ';'.join(templ.split(',')))
                        try:
                            # print(input)
                            outputs.append(generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))
                        except KeyError:
                            outputs.append(None)
                    if len([output for output in outputs if output])==0:
                        p[l] = None
                        continue
                    if WeightInVote:
                        p[l] = weight_vote(outputs,
                                           [relations_distrust.get(relation, None) for relation in relations],
                                           [labels_distrust.get(relation.split()[0], None) for relation in relations])
                    else:
                        p[l] = vote(outputs)
            for l in d:
                print("%s\t%s" % (d[l],l), file=f)
            for l in p:
                print("%s\t%s" % (p[l], l), file=f)
            print('', file=f)
            predicts.append(p)

    corrects = 0
    tot = 0
    for i, d in enumerate(predicts):
        for l in d:
            tot += 1
            if language == 'he' and d[l] and answers[i][l]:
                d[l] = d[l].replace('ף', 'פ').replace('ץ', 'צ').replace('ך', 'כ').replace('ם', 'מ').replace('ן', 'נ')
                answers[i][l] = answers[i][l].replace('ף', 'פ').replace('ץ', 'צ').replace('ך', 'כ').replace('ם', 'מ').replace('ן', 'נ')
            if d[l] == answers[i][l]:
                corrects += 1
        # print(i, corrects/tot)
    print(corrects/tot)
    print(write_path, 'written')
    return corrects/tot


def read_wrap(data_fn, ans_fn):
    data = readtestdata(data_fn)
    answers = readtestdata(ans_fn)

    # length = Counter([len(d) for d in answers]).most_common()[0][0]
    # mask = [len(d) == length for d in answers]
    length = max([len(d) for d in answers])
    mask = [len(d) >= length / 2 for d in answers]

    data = [d for i, d in enumerate(data) if mask[i]]
    answers = [d for i, d in enumerate(answers) if mask[i]]

    return data, answers

def lemma_read_wrap(data_fn, ans_fn):
    data = lemma_readtestdata(data_fn)
    answers = lemma_readtestdata(ans_fn)

    return data, answers

if __name__=='__main__':
    import os

    meta = exp_dir + f'_{scoring_threshold}'
    model_path = os.path.join(model_dir_path, f'{meta}_model')
    write_path = os.path.join(model_dir_path, f'{meta}_from_lemma_output.txt')

    print('testing inflection from lemma')
    print('model path:', model_path)
    print('writing to:', write_path)

    data, answers = lemma_read_wrap(os.path.join(inflec_data_dir, f'{language}.um.{paradigm}.lemma.txt'),
                              os.path.join(inflec_data_dir, f'{language}.um.{paradigm}.lemma_paradigms.txt'))
    int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % model_path, "rb"))

    init()
    model.populate(model_path)

    test(data, answers, write_path, None, None)


# #    global int2char, char2int, VOCAB_SIZE, model
#     data, answers = read_wrap(argv[1], argv[2])
#     # data = readtestdata(argv[1])
#     # answers = readtestdata(argv[2])
#     model_path = argv[3]+'_model'
#     int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % model_path,
#                                                       "rb"))
#     init()
#     model.populate(model_path)
#     all_labels = list(answers[0].keys())
#     accuracy = test(data, all_labels, answers, argv[3]+'_output.txt')
