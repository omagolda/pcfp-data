from sys import stdout, argv
from collections import Counter

import dynet as dy
import random
import pickle

WeightInVote = True

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

def vote(outputs):
    return Counter(outputs).most_common()[0][0]

def weight_vote(outputs, relation_distrusts, labels_distrust):
    # assert len(outputs)==len(distrusts)
    count = Counter(outputs)
    _ = count.pop(None, None)

    if count.most_common()[0][1] != 1:
        return Counter(outputs).most_common()[0][0]

    elif set(relation_distrusts) != {None}:
        rds = [d for d in relation_distrusts if d is not None]
        return outputs[relation_distrusts.index(min(rds))]

    elif set(labels_distrust) != {None}:
        lds = [d for d in labels_distrust if d is not None]
        return outputs[labels_distrust.index(min(lds))]

    else:
        return random.choice(outputs)

def test(partial_data, all_labels, answers, write_path, relations_distrust, labels_distrust, covered_labels=None):
    with open(write_path, 'w', encoding='utf8') as f:
        for d in partial_data:
            dy.renew_cg()
            forms = [[c for c in wf] + ['+'] + l.split(',')
                     for l,wf in d.items() if wf != None]
            labels = [';'.join(l.split(',')) for l,wf in d.items() if wf != None]
            if covered_labels and any([l not in covered_labels for l in labels]):
                continue

            for l in all_labels:
                # if d[l] == None:
                if d.get(l, None) == None:
                    if covered_labels and ';'.join(l.split(',')) not in covered_labels:
                        continue
                    inputs = [f + ['+'] + l.split(',') for f in forms]
                    relations = []
                    outputs = []
                    for i, input in enumerate(inputs):
                        try:
                            outputs.append(generate(input, enc_fwd_lstm, enc_bwd_lstm, dec_lstm))
                            relations.append(labels[i]+' '+';'.join(l.split(',')))
                        except KeyError:
                            continue
                    if len(outputs)<=0:
                        continue
                    if WeightInVote:
                        d[l] = weight_vote(outputs,
                                           [relations_distrust.get(relation, None) for relation in relations],
                                           [labels_distrust.get(relation.split()[0], None) for relation in relations])
                    else:
                        d[l] = vote(outputs)
            for l in d:
                print("%s\t%s" % (d[l],l), file=f)
            print('', file=f)

    corrects = 0
    tot = 0
    for i, d in enumerate(partial_data):
        for l in d:
            tot += 1
            if d[l] == answers[i][l]:
                corrects += 1
        print(i, corrects/tot)
    print(corrects/tot)
    print(write_path, 'written')
    return corrects/tot


def read_wrap(data_fn, ans_fn):
    data = readtestdata(data_fn)
    answers = readtestdata(ans_fn)

    length = Counter([len(d) for d in answers]).most_common()[0][0]
    mask = [len(d) == length for d in answers]

    data = [d for i, d in enumerate(data) if mask[i]]
    answers = [d for i, d in enumerate(answers) if mask[i]]

    return data, answers


if __name__=='__main__':
#    global int2char, char2int, VOCAB_SIZE, model
    data, answers = read_wrap(argv[1], argv[2])
    # data = readtestdata(argv[1])
    # answers = readtestdata(argv[2])
    model_path = argv[3]+'_model'
    int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % model_path,
                                                      "rb"))
    init()
    model.populate(model_path)
    all_labels = list(answers[0].keys())
    accuracy = test(data, all_labels, answers, argv[3]+'_output.txt')
