# Import libraries

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import keras as kr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import collections
import hashlib
import json
import warnings

# Function

def text_to_word_sequence(
    input_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ",
):
    if lower:
        input_text = input_text.lower()

    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    input_text = input_text.translate(translate_map)

    seq = input_text.split(split)
    return [i for i in seq if i]

class Tokenizer(object):

    def __init__(
        self,
        num_words=None,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=" ",
        char_level=False,
        oov_token=None,
        analyzer=None,
        **kwargs
    ):
        # Legacy support
        if "nb_words" in kwargs:
            warnings.warn(
                "The `nb_words` argument in `Tokenizer` "
                "has been renamed `num_words`."
            )
            num_words = kwargs.pop("nb_words")
        document_count = kwargs.pop("document_count", 0)
        if kwargs:
            raise TypeError("Unrecognized keyword arguments: " + str(kwargs))

        self.word_counts = collections.OrderedDict()
        self.word_docs = collections.defaultdict(int)
        self.filters = filters
        self.split = split
        self.lower = lower
        self.num_words = num_words
        self.document_count = document_count
        self.char_level = char_level
        self.oov_token = oov_token
        self.index_docs = collections.defaultdict(int)
        self.word_index = {}
        self.index_word = {}
        self.analyzer = analyzer

    def fit_on_texts(self, texts):
        for text in texts:
            self.document_count += 1
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = text_to_word_sequence(
                        text,
                        filters=self.filters,
                        lower=self.lower,
                        split=self.split,
                    )
                else:
                    seq = self.analyzer(text)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                # In how many documents each word occurs
                self.word_docs[w] += 1

        wcounts = list(self.word_counts.items())
        wcounts.sort(key=lambda x: x[1], reverse=True)
        # forcing the oov_token to index 1 if it exists
        if self.oov_token is None:
            sorted_voc = []
        else:
            sorted_voc = [self.oov_token]
        sorted_voc.extend(wc[0] for wc in wcounts)

        # note that index 0 is reserved, never assigned to an existing word
        self.word_index = dict(
            zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))
        )

        self.index_word = {c: w for w, c in self.word_index.items()}

        for w, c in list(self.word_docs.items()):
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        self.document_count += len(sequences)
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                self.index_docs[i] += 1

    def texts_to_sequences(self, texts):
        return list(self.texts_to_sequences_generator(texts))

    def texts_to_sequences_generator(self, texts):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for text in texts:
            if self.char_level or isinstance(text, list):
                if self.lower:
                    if isinstance(text, list):
                        text = [text_elem.lower() for text_elem in text]
                    else:
                        text = text.lower()
                seq = text
            else:
                if self.analyzer is None:
                    seq = text_to_word_sequence(
                        text,
                        filters=self.filters,
                        lower=self.lower,
                        split=self.split,
                    )
                else:
                    seq = self.analyzer(text)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if num_words and i >= num_words:
                        if oov_token_index is not None:
                            vect.append(oov_token_index)
                    else:
                        vect.append(i)
                elif self.oov_token is not None:
                    vect.append(oov_token_index)
            yield vect

    def sequences_to_texts(self, sequences):
        return list(self.sequences_to_texts_generator(sequences))

    def sequences_to_texts_generator(self, sequences):
        num_words = self.num_words
        oov_token_index = self.word_index.get(self.oov_token)
        for seq in sequences:
            vect = []
            for num in seq:
                word = self.index_word.get(num)
                if word is not None:
                    if num_words and num >= num_words:
                        if oov_token_index is not None:
                            vect.append(self.index_word[oov_token_index])
                    else:
                        vect.append(word)
                elif self.oov_token is not None:
                    vect.append(self.index_word[oov_token_index])
            vect = " ".join(vect)
            yield vect

    def texts_to_matrix(self, texts, mode="binary"):
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode="binary"):
        if not self.num_words:
            if self.word_index:
                num_words = len(self.word_index) + 1
            else:
                raise ValueError(
                    "Specify a dimension (`num_words` argument), "
                    "or fit on some text data first."
                )
        else:
            num_words = self.num_words

        if mode == "tfidf" and not self.document_count:
            raise ValueError(
                "Fit the Tokenizer on some data before using tfidf mode."
            )

        x = np.zeros((len(sequences), num_words))
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            counts = collections.defaultdict(int)
            for j in seq:
                if j >= num_words:
                    continue
                counts[j] += 1
            for j, c in list(counts.items()):
                if mode == "count":
                    x[i][j] = c
                elif mode == "freq":
                    x[i][j] = c / len(seq)
                elif mode == "binary":
                    x[i][j] = 1
                elif mode == "tfidf":
                    # Use weighting scheme 2 in
                    # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
                    tf = 1 + np.log(c)
                    idf = np.log(
                        1
                        + self.document_count / (1 + self.index_docs.get(j, 0))
                    )
                    x[i][j] = tf * idf
                else:
                    raise ValueError("Unknown vectorization mode:", mode)
        return x


#Working with data

data = pd.read_excel("summarizer/Data/news.xlsx")
data.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)

## Splitting the data
short = data['Short']
summary = data['Headline']

# Since < and > from default tokens cannot be removed
filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

# for decoder sequence
summary = summary.apply(lambda x: '<go> ' + x + '<stop>')

short_tokenizer = Tokenizer(oov_token=oov_token)
summary_tokenizer = Tokenizer(filters=filters, oov_token=oov_token)

short_tokenizer.fit_on_texts(short)
summary_tokenizer.fit_on_texts(summary)

inputs = short_tokenizer.texts_to_sequences(short)
targets = summary_tokenizer.texts_to_sequences(summary)

# Vocab size
encoder_vocab_size = len(short_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1

"""Finding max length
Taking values >= to the 75th percentile by rounding off"""

# short_len = pd.Series([len(x) for x in short])
# summary_len = pd.Series([len(x) for x in summary])

# print(short_len.describe())
"""
count    55104.000000
mean       368.003049
std         26.235510
min        280.000000
25%        350.000000
50%        369.000000
75%        387.000000
max        469.000000

"""

print(">>>>>>>>>>>>>")

# print(summary_len.describe())
"""
count    55104.000000
mean        62.620282
std          7.267463
min         19.000000
25%         58.000000
50%         62.000000
75%         68.000000
max         95.000000
"""


encoder_max_len = 400
decoder_max_len = 75

# Truncating sequences for identical sequence lengths
inputs = kr.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_max_len, padding='post', truncating='post')
targets = kr.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_max_len, padding='post', truncating='post')

# Creating dataset pipeline

inputs = tf.cast(inputs, dtype=tf.int32)
targets = tf.cast(targets, dtype=tf.int32)

BUFFER_SIZE = 20_000
BATCH_SIZE = 64

dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Positional Encoding for adding notion of position among words as unlike RNN this is non-directional

def get_angles(position, i,  d_model):
    angle_rates = 1/ np.power(10_000, (2*(i//2))/ np.float32(d_model))
    return position*angle_rates

def positional_encoding(position, d_model):
    angle_radians = get_angles(
        np.arange(position)[:,np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    
    # apply sin to even indices (2i) and cos to odd (2i+1)
    angle_radians[:,0::2] = np.sin(angle_radians[:,0::2])
    angle_radians[:,1::2] = np.sin(angle_radians[:,1::2])
    
    position_encoding = angle_radians[np.newaxis, ...]
    
    return tf.cast(position_encoding, dtype=tf.float32)

"""
MASKING":

1. Padding mask.
2. Lookahead mask.
"""

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_lookahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

# Building the model

## Scaled Dot Product
def scaled_dot_prod_attention(q, k, v, mask):
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk/tf.math.sqrt(dk)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    attention_weights = tf.nn.softmax(scaled_attention_logits, v)
    
    output = tf.matmul(attention_weights, v)
    return output, attention_weights

# Multi-Headed Attention

class MultiHeadAttention(kr.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        
        self.depth = d_model // self.num_heads
        
        self.wq = kr.layers.Dense(d_model)
        self.wk = kr.layers.Dense(d_model)
        self.wv = kr.layers.Dense(d_model)
        
        self.dense = kr.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weight = scaled_dot_prod_attention(q, k, v, mask)
        
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])
        
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output, attention_weight
    
# Feed Forward Network
def point_wise_feed_forward(d_model, dff):
    return kr.Sequential([
        kr.layers.Dense(dff, activation='relu'),
        kr.layers.Dense(d_model)
    ])

## Fundamental Unit of transformer encoder

class EncodeLayer(kr.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncodeLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        
        self.norm1 = kr.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = kr.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = kr.layers.Dropout(rate)
        self.dropout2 = kr.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training)
        out1 = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2
    
## Fundamental Unit of transformer decoder

class DecodeLayer(kr.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecodeLayer, self).__init__()
        
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward(d_model, dff)
        
        self.norm1 = kr.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = kr.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = kr.layers.LayerNormalization(epsilon=1e-6)
        
        self.drop1 = kr.layers.Dropout(rate)
        self.drop2 = kr.layers.Dropout(rate)
        self.drop3 = kr.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights_block1 = self.mha1(x,x,x,look_ahead_mask)
        attn1 = self.drop1(attn1, training)
        out1 = self.norm1(attn1 + x)
        
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.drop2(attn2, training)
        out2 = self.norm2(attn2 + out1)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.drop3(ffn_output, training)
        out3 = self.norm3(ffn_output + out2)
        
        return out3, attn_weights_block1, attn_weights_block2
    
# Encoder consisting of Multiple EncoderLayer(s)

class Encoder(kr.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_positioning_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embed = kr.layers.Embedding(input_vocab_size, d_model)
        self.pos_enc = positional_encoding(maximum_positioning_encoding, self.d_model)
        
        self.enc_layers = [EncodeLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = kr.layers.Dropout(rate)
        
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_enc[:, :seq_len, :]
        
        x = self.dropout(x, training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
            
        return x
    
# Decoder consisting of multiple decoder layer(s)

class Decoder(kr.layers.Layer):
    def __init__(self, num_layers ,d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embed = kr.layers.Embedding(target_vocab_size, d_model)
        self.pos = positional_encoding(maximum_position_encoding, d_model)
        
        self.dec_layers = [DecodeLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.drop = kr.layers.Dropout(rate)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weigh = {}
        
        x = self.embed(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos[: , :seq_len, :]
        
        x = self.drop(x, training)
        
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)
            
            attention_weigh['decoder_layer{}_block1'. format(i+1)] = block1
            attention_weigh['decoder_layer{}_block2'. format(i+1)] = block2
            
        return x, attention_weigh
    

# Transformer

class Transformer(kr.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        
        self.final_layer = kr.layers.Dense(target_vocab_size)
        
    def call(self, inp, tar, training, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)
        
        final_output = self.final_layer(dec_output)
        
        return final_output, attention_weights
    
