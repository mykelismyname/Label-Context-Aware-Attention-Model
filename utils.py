import os
import re
import json
import stanza
import spacy
import ast
import numpy as np
from glob import glob

st_nlp = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000)
sp_nlp = spacy.load('en_core_web_sm')
#creating a directory for the plots
def create_directories_per_series_des(name=''):
    _dir = os.path.abspath(os.path.join(os.path.curdir, name))
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    return _dir

def create_dataset_to_encode(datafiles, tokenizer, output_dir):
    with open(os.path.join(output_dir, 'ebm_comet_train_dev'), 'w') as t:
        for file in datafiles:
            file_open = open(file, 'r')
            sentence = []
            for line in file_open.readlines():
                line = line.strip()
                if (line.startswith("[['") and line.endswith("']]")) or \
                        (line.startswith("['") and line.endswith("']")) or re.search('\[\]', line):
                    pass
                else:
                    if line == '\n' or line == '':
                        if sentence:
                            sentence_strings = ' '.join(i for i in sentence)
                            toks_split = tokenize(sentence_strings.strip().lower(), tokenizer)
                            t.write(' '.join(toks_split))
                            t.write(' ')
                        sentence.clear()
                    else:
                        line = re.split(' ', line, maxsplit=2)
                        line_ = line[0]
                        sentence.append(line_.strip())
        t.close()

def tokenize(sent, tokenizer):
    k = []
    if tokenizer.lower() == 'stanford':
        doc = st_nlp(sent)
        for s in doc.sentences:
            for t in s.tokens:
                k.append(t.text)
    elif tokenizer.lower() == 'spacy':
        doc = sp_nlp(sent)
        for s in doc:
            k.append(s.text)
    return k



def main(data_input, tokenizer):
    word_map = {}
    for file in data_input:
        file_sentences, file_sentence_labels = [], []
        with open(file, 'r') as f:
            file_lines = f.readlines()
            sentence = []
            for line in file_lines:
                line = line.strip()
                if (line.startswith("[['") and line.endswith("']]")) or \
                        (line.startswith("['") and line.endswith("']")) or re.search('\[\]', line):
                    multilabel = ast.literal_eval(line)
                    if not multilabel:
                        multilabel.append('No-label')
                    elif all(type(i) == list for i in multilabel):
                        multilabel = [i for j in multilabel for i in j]
                    multilabel = list(set(multilabel))
                    file_sentence_labels.append(multilabel)
                else:
                    if line == '\n' or line == '':
                        if sentence:
                            toks_split, tags_split = re_split_sentences(sentence, tokenizer=tokenizer)
                            file_sentences.append(list(zip(toks_split, tags_split)))
                        sentence.clear()
                    else:
                        line = re.split(' ', line,  maxsplit=2)
                        if len(line) > 2:
                            line[-1] = ast.literal_eval(line[-1])
                            token_label = list(set(line[-1]))
                            if len(token_label) > 1:
                                pass
                            if line[1][0] in ['B', 'I']:
                                line[1] = line[1][:2]+token_label[-1]
                            else:
                                if line[1].__contains__('outcome'):
                                    line[1] = line[1][:2]
                                else:
                                    pass
                        sentence.append((line[0].strip(), line[1]))
            f.close()

#used for converting embeddings in text to byte
def text2embed(emb_file, emb_size, dest_dir):
    output_file = os.path.basename(emb_file).split('.')
    with open(emb_file, 'r') as e_f, open(os.path.join(dest_dir, '{}.npy'.format(output_file[0])), 'wb') as l_f:
        i = 0
        e_f_lines = e_f.readlines()
        labels = np.zeros((len(e_f_lines), emb_size))
        for n in e_f_lines:
            n = n.split()
            n = [float(i) for i in n[1:]]
            print(len(n))
            labels[i] = n
            i+=1
            print(len(n))
        np.save(l_f, labels)

#append new words to the top of the embeddings text file for LSAN method
def addWordEmbs(words, emb_size, embs_file):
    embs_file_dir = os.path.dirname(embs_file)
    embs_file_name = os.path.basename(embs_file)
    dummy_file = embs_file_dir+'/'+embs_file_name.split('.')[0]+'.bak'
    word_embs = np.zeros(emb_size)
    with open(embs_file, 'r') as e_f, open(dummy_file, 'w') as d_f:
        for word in words:
            d_f.write(word+' '+' '.join(str(i) for i in word_embs))
            d_f.write('\n')
        for line in e_f:
            d_f.write(line)
    os.remove(embs_file)
    os.rename(dummy_file, embs_file_dir+'/'+embs_file_name)

def createVocabFromGloveWordEmbs(vectors):
    dest = os.path.dirname(vectors)
    voc_file, num = {}, 0
    with open(vectors, 'r') as vec, open(dest+'/token_labels.json','w') as voc:
        for i in vec:
            d = i.split()
            voc_file[d[0].strip()] = num
            num += 1
        json.dump(voc_file, voc, indent=2)
        vec.close()
        voc.close()

def algin_vocab_embeddings(vocab_file, embs_file):
    embs_file_dir = os.path.dirname(embs_file)
    embs_file_name = os.path.basename(embs_file)
    dummy_file = embs_file_dir + '/' + embs_file_name.split('.')[0] + '.bak'
    with open(vocab_file, 'r') as f, open(embs_file, 'r') as g, open(dummy_file, 'w') as h:
        vocab = json.load(f)
        vocab = dict(sorted(vocab.items(), key=lambda x:x[1]))
        embeddings = g.readlines()
        for word,vec in vocab.items():
            for i,line in enumerate(embeddings):
                line = line.split()
                if word == line[0].strip():
                    h.write(word.strip() + ' ' + ' '.join(str(i) for i in line[1:]))
                    h.write('\n')
                    break
        f.close()
    os.remove(embs_file)
    os.rename(dummy_file, embs_file_dir + '/' + embs_file_name)

def count(embs_file):
    with open(embs_file, 'r') as g:
        embeddings = g.readlines()
        l = []
        for i,line in enumerate(embeddings):
            line = line.split()
            l.append(line[0])
            print(i, line[0])
        print(len(l))
        g.close()

# data = 'multi-label-module/multi-labelled-data/'
# data = [i for i in glob('{}/*.txt'.format(data)) if i.__contains__('train') or i.__contains__('dev')]
# print(data)
# #create_dataset_to_encode(datafiles=data, tokenizer='spacy')
# main(data, tokenizer='spacy')
# #vocab_count('ebm_comet_train_dev')
# text2embed('multi-labelled-data/lwan_ebm_comet/lwan/token_labels_embed.txt', 300, 'multi-labelled-data/lwan_ebm_comet/lwan/')
# addWordEmbs(['<pad>','<unk>'], 300, 'multi-label-module/multi-labelled-data/LSAN/word_embed.txt')
# createVocabFromGloveWordEmbs('multi-labelled-data/LWAN_1/token_labels_embed.txt')
# algin_vocab_embeddings('multi-labelled-data/lwan_ebm_comet/lwan/token_labels.json', 'multi-labelled-data/lwan_ebm_comet/lwan/token_labels_embed.txt')
# count('multi-labelled-data/lwan2/word_embed.txt')