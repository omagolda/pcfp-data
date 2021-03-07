import train_n_gt_1
import pickle
from main_n_gt_1 import pack_params
import dynet as dy
# import test_n_gt_1

EOS = "<EOS>"

int2char = [EOS,'+']
char2int = {EOS:0,'+':1}

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 100
STATE_SIZE = 100
ATTENTION_SIZE = 100


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



def test(test_data, write_path):
    corrects = 0
    with open(write_path, 'w', encoding='utf8') as f:
        for i, d in enumerate(test_data):
            dy.renew_cg()
            output = generate(d[0], enc_fwd_lstm, enc_bwd_lstm, dec_lstm)
            f.write('\t'.join(d[0].split('+') + [output]) + '\n')

            if output == d[1]:
                corrects += 1
    print(corrects / len(test_data))
    print(write_path, 'written')
    return corrects / len(test_data)


if __name__ == '__main__':
    lang = 'heb'
    train_path = ''
    test_path = ''
    model_path = ''
    write_path = ''

    train_data = train_n_gt_1.my_readdata(train_path, return_added=False)
    train_n_gt_1.init()
    train_n_gt_1.train(train_n_gt_1.model, train_data)
    train_n_gt_1.model.save(model_path)
    pickle.dump((train_n_gt_1.int2char,train_n_gt_1.char2int,train_n_gt_1.VOCAB_SIZE),
                open("%s.obj.pkl" % model_path, "wb"))

    test_data = train_n_gt_1.my_readdata(test_path, return_added=False)
    # int2char, char2int, VOCAB_SIZE = pickle.load(open("%s.obj.pkl" % model_path, "rb"))
    int2char, char2int, VOCAB_SIZE = train_n_gt_1.int2char,train_n_gt_1.char2int,train_n_gt_1.VOCAB_SIZE
    list_of_stuff = pack_params(train_n_gt_1)
    init_existing(list_of_stuff + [int2char, char2int])
    accuracy = test(test_data, write_path)



