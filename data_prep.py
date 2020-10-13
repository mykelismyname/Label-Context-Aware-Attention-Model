import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import numpy as np
import re
import os
import ast
import pandas as pd
import argparse
import json
import utils as utils
import stanza
from glob import glob
from gensim.models import Word2Vec

def create_vocabularly(data_input, output_dir, tokenizer):
    #define a tokenizer borrowing keras processing
    vocab, tag_count, tag_map, all_sentences = {"<unk>":0, "<pad>":1}, {}, {"<pad>":0}, {}
    num_words, num_tags = 2, 1
    all_sentences, all_sentence_labels = {}, []

    if type(data_input) == str:
        data_input = [data_input]
    '''
    creating a vocabularly that merges the training and validation data otherwise (else block) only the training data
    steps include 1) creating a list of all sentences in the datasets. 2) creating a dictionary of all the words and all labels 
    '''
    print('data input', data_input)
    with open(os.path.join(output_dir, 'ebm_comet_train_dev'), 'w') as tk, open(os.path.join(output_dir, 'ebm_comet_train_dev_tags'), 'w') as tg:
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
                                toks_split = [i.lower() for i in toks_split]
                                file_sentences.append(list(zip(toks_split, tags_split)))
                                tk.write(' '.join(toks_split))
                                tk.write(' ')
                                tg.write(' '.join(tags_split))
                                tg.write(' ')
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

            for sent in file_sentences:
                for feature in sent:
                    word, tag = feature
                    if word not in vocab:
                        vocab[word] = num_words
                        num_words += 1
                    if tag not in tag_map:
                        tag_map[tag] = num_tags
                        tag_count[tag] = 1
                        num_tags += 1
                    else:
                        tag_count[tag] += 1
            all_sentences[file] = [file_sentences, file_sentence_labels]
            all_sentence_labels += file_sentence_labels

    print(len(all_sentences), len(all_sentence_labels))
    index2word = dict([(v,k) for k,v in vocab.items()])
    index2tag = dict([(v,k) for k,v in tag_map.items()])

    with open(os.path.join(output_dir, 'vocab_.json'), 'w') as word_map,\
            open(os.path.join(output_dir, 'token_labels_.json'), 'w') as tok, \
                open(os.path.join(output_dir, 'sentence_labels_.txt'), 'w') as slabel:
        json.dump(vocab, word_map, indent=2)
        json.dump(dict(sorted(tag_map.items(), key=lambda x:x[0])), tok, indent=2)
        all_sentence_labels = sorted(list(set([l for sent_labels in all_sentence_labels for l in sent_labels])))
        for l in all_sentence_labels:
            slabel.write('{}\n'.format(l))
        word_map.close()
        tok.close()
        slabel.close()
    return all_sentences, vocab, index2word, num_words, tag_map, index2tag

#re-tokenize tokens usng spacy tokenizer
def re_split_sentences(s, tokenizer):
    toks = [i[0] for i in s]
    toks_joined = ' '.join(toks)
    tags = [i[1] for i in s]
    toks_split = utils.tokenize(toks_joined, tokenizer)
    tags_split = []
    m, n, x_ = 0, 0, ''
    v = False

    for y,x in enumerate(toks_split):
        x = x.strip()
        if y == n:
            if x == toks[m]:
                tags_split.append(tags[m])
                m += 1
            else:
                x_ += x
                if tags[m][0] in ['B', 'I']:
                    tags_split.append(tags[m])
                    for j in toks_split[y+1:]:
                        tags_split.append('I-{}'.format(tags[m][2:]))
                        x_ += j
                        n += 1
                        if x_ == toks[m]:
                            m += 1
                            x_ = ''
                            break

                elif tags[m][0].lower() in ['e', 's']:
                    tags_split.append(tags[m-1])
                    for j in toks_split[y+1:]:
                        tags_split.append(tags[m-1])
                        x_ += j
                        n += 1
                        if x_ == toks[m]:
                            m += 1
                            x_ = ''
                            tags_split[-1] = tags[m]
                            break
                    v = True
                else:
                    tags_split.append(tags[m])
                    for j in toks_split[y+1:]:
                        tags_split.append(tags[m])
                        x_ += j
                        n += 1
                        if x_ == toks[m]:
                            m += 1
                            x_ = ''
                            break
            n += 1
    if v:
        print(toks_split)
        print(tags_split)
    if len(toks_split) != len(tags_split):
        print(len(toks_split), len(tags_split))
        print('\ncheck these\n{}\n{}\n{}'.format(toks_split, toks, tags_split))
        raise ValueError('Something is wrong')
    return toks_split, tags_split

#preparing data for LSAN model
def pre_process_(documents, vocab, token_label_vocab, output_dir, method=None):
    for file,docs in documents.items():
        docs, sentence_labels = docs
        document_matrix = []
        token_label_matrix = []
        sentence_unique_labels = sorted(list(set([l for labs in sentence_labels for l in labs])))
        sentence_label_matrix = np.zeros((len(docs), len(sentence_unique_labels)))
        print(sentence_unique_labels)
        sentence_label_map = dict([(sentence_unique_labels[index], index) for index in range(len(sentence_unique_labels))])
        print(sentence_label_map)
        print(token_label_vocab)
        i = 0
        for doc,lab in zip(docs, sentence_labels):
            doc_matrix = np.zeros(len(doc))
            tok_matrix = np.zeros(len(doc))
            for k,d in enumerate(doc):
                word, tag = d
                doc_matrix[k] = vocab[word]
                if method.lower() == 'lwan':
                    tok_matrix[k] = token_label_vocab[tag]
            for l in lab:
                sentence_label_matrix[i][sentence_label_map[l]] = 1
            i += 1
            document_matrix.append(doc_matrix.tolist())
            token_label_matrix.append(tok_matrix.tolist())


        file_name = os.path.basename(file).split('.')
        document_matrix, token_label_matrix = np.array(document_matrix), np.array(token_label_matrix)
        for b,matrix in enumerate([(np.array(document_matrix), np.array(token_label_matrix)), sentence_label_matrix]):
            if b == 0:
                initial = 'X'
                matrix,token_matrix = matrix
            else:
                initial = 'Y'
            if method.lower() == 'lsan':
                with open(os.path.join(output_dir, '{}_{}.npy'.format(initial, file_name[0])), 'wb') as file_out:
                    np.save(file_out, matrix)
            elif method.lower() == 'lwan':
                with open(os.path.join(output_dir, '{}_{}.npy'.format(initial, file_name[0])), 'wb') as file_out,\
                        open(os.path.join(output_dir, '{}_label_{}.npy'.format(initial, file_name[0])), 'wb') as file_tok_out:
                    np.save(file_out, matrix)
                    np.save(file_tok_out, token_matrix)
        print('No of sentences in {} set is {}'.format(file_name, len(docs)), document_matrix.shape, sentence_label_matrix.shape)

    return document_matrix, token_label_matrix, sentence_label_matrix

#preparing data for LSAN model
def pre_process(documents, vocab, token_label_vocab, output_dir, method=None, pad_value=500):
    for file,docs in documents.items():
        docs, sentence_labels = docs
        document_matrix = np.ones((len(docs), pad_value)).astype(int)
        token_label_matrix = np.zeros((len(docs), pad_value)).astype(int)
        sentence_unique_labels = sorted(list(set([l for labs in sentence_labels for l in labs])))
        sentence_label_matrix = np.zeros((len(docs), len(sentence_unique_labels)))
        print(sentence_unique_labels)
        sentence_label_map = dict([(sentence_unique_labels[index], index) for index in range(len(sentence_unique_labels))])
        print(sentence_label_map)
        print(token_label_vocab)
        i = 0
        for doc,lab in zip(docs, sentence_labels):
            if len(doc) > pad_value:
                raise ValueError('Youve got sentences longer than 500')
            for k,d in enumerate(doc):
                word, tag = d
                document_matrix[i][k] = vocab[word]
                if method.lower() == 'lwan':
                    if i < 5:
                        print(word, tag, vocab[word], token_label_vocab[tag])
                    token_label_matrix[i][k] = token_label_vocab[tag]
            for l in lab:
                sentence_label_matrix[i][sentence_label_map[l]] = 1
            i += 1
            # print('\n')

        file_name = os.path.basename(file).split('.')
        for b,matrix in enumerate([(document_matrix, token_label_matrix), sentence_label_matrix]):
            if b == 0:
                initial = 'X'
                matrix,token_matrix = matrix
            else:
                initial = 'Y'
            if method.lower() == 'lsan':
                with open(os.path.join(output_dir, '{}_{}.npy'.format(initial, file_name[0])), 'wb') as file_out:
                    np.save(file_out, matrix)
            elif method.lower() == 'lwan':
                with open(os.path.join(output_dir, '{}_{}.npy'.format(initial, file_name[0])), 'wb') as file_out,\
                        open(os.path.join(output_dir, '{}_label_{}.npy'.format(initial, file_name[0])), 'wb') as file_tok_out:
                    np.save(file_out, matrix)
                    np.save(file_tok_out, token_matrix)
        print('No of sentences in {} set is {}'.format(file_name, len(docs)), document_matrix.shape, sentence_label_matrix.shape)

    return document_matrix, token_label_matrix, sentence_label_matrix

#solitting entire dataset into train, dev and test sets
def create_train_dev_test(file, split_percentage):
    def write_to_file(file, sents):
        with open(file, 'w') as w:
            for sent_list in sents:
                for line in sent_list:
                    w.write('{}\n'.format(line))
                w.write('\n')
            w.close()

    with open(file, 'r') as f:
        file_lines = f.readlines()
        sentences, sentence = [], []
        for line in file_lines:
            line = line.strip()
            if line == '\n' or line == '':
                if sentence:
                    sentences.append([i for i in sentence])
                sentence.clear()
            else:
                sentence.append(line)
        f.close()
    print('dataset length', len(sentences))
    np.random.shuffle(sentences)

    if len(split_percentage) == 2:
        dataset_lengh = len(sentences)
        train_length, test_length = [int((i/100)*dataset_lengh) for i in split_percentage]
        file_path = os.path.dirname(file)
        for i in range(3):
            if i == 0:
                train_sents = sentences[:train_length]
                write_to_file(file=file_path+'/train.txt', sents=train_sents)
                print('len of train sents', len(train_sents))
            elif i == 1:
                test_sents = sentences[train_length:(train_length+test_length)]
                write_to_file(file=file_path + '/test.txt', sents=test_sents)
                print('len of test sents',len(test_sents))
            else:
                dev_sents = sentences[(train_length+test_length):]
                write_to_file(file=file_path + '/dev.txt', sents=dev_sents)
                print('len of dev sents',len(dev_sents))
    else:
        raise ValueError('You might want to pass a list with both train and test percentages, not dev percentage = 100 -(train+test)')

def vocab_count(file):
    word_map = {}
    with open(file,'r') as f:
        for i in f.readlines():
            i = re.split(' ', i)
            for j in i:
                if j not in word_map:
                    word_map[j] = 1
                else:
                    word_map[j] += 1
    file_path = os.path.dirname(os.path.abspath(file))
    word_map = dict(sorted(word_map.items(), key=lambda x:x[1], reverse=True))
    with open(os.path.join(file_path, 'vocab.json'), 'w') as d:
        json.dump(word_map, d, indent=2)
        d.close()

def createOneHotVector(vocab):
    one_hot_vocab = np.zeros((len(vocab), len(vocab)))
    vocab_ = {}
    for i,j in vocab.items():
        one_hot_vocab[j][j] = 1
        vocab_[i] = one_hot_vocab[j][j]
    return one_hot_vocab

def count_sentence_length(files):
    all_sentences = []
    for i in files:
        a = open(i, 'r')
        sentence = []
        for t in a.readlines():
            t = t.strip()
            if t == '\n' or t == '':
                if sentence:
                    all_sentences.append([i for i in sentence])
                sentence.clear()
            else:
                t.split()
                sentence.append(t[0])
        a.close()
    longest_sentence = max([len(i) for i in all_sentences])
    average_sent_len = np.mean([len(i) for i in all_sentences])
    print('Longest sentence length', longest_sentence)
    print('Average sentence length', average_sent_len)
    # return longest_sentence, average_sent_len

def convert_json_to_text(file):
    file_dir = os.path.dirname(file)
    file_name = os.path.basename(file).split('.')
    with open(file, 'r') as a, open('{}/{}.txt'.format(file_dir, file_name[0]), 'w') as b:
        for i in json.load(a):
            b.write(i)
            b.write('\n')
        a.close()
        b.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data source')
    parser.add_argument('--dataset', type=str, help='either train, test or dev: Used when preparing transformer data')
    parser.add_argument('--outputdir', type=str, help='directory to hold output files')
    parser.add_argument('--tokenizer', default='spacy', type=str, help='spacy or stanford for tokenization')
    parser.add_argument("--function", action='store_true', help="e.g create training, validation and test data")
    parser.add_argument("--function_name", type=str, help="e.g count_sentence_length or create_train_dev_test")
    parser.add_argument("--split_percentage", type=str, help="[80, 10], 60% train, 10% test 100-(80+10) dev")
    parser.add_argument("--vocab", type=str, help="vocabularly source")
    parser.add_argument("--token_vocab", type=str, help="vocabularly source")
    parser.add_argument("--method", type=str, default=None, help="LSAN")

    args = parser.parse_args()
    if args.outputdir:
        utils.create_directories_per_series_des(args.outputdir)
    if args.function:
        if args.function_name.strip() == 'create_train_dev_test':
            file = glob('{}/*comet_nlp.txt'.format(args.data))
            split_percentage = ast.literal_eval(args.split_percentage)
            create_train_dev_test(file=file[0], split_percentage=split_percentage)
        elif args.function_name.strip() == 'count_sentence_length':
            files = glob('{}/*.txt'.format(args.data))
            print(files)
            count_sentence_length(files)
        elif args.function_name.strip() == 'convert_json_to_text':
            convert_json_to_text(args.data)
        elif args.function_name.strip() == 'transformer_data_preparation':
            X = np.load("{}/X_{}.npy".format(args.data, args.dataset))
            Y = np.load("{}/X_label_{}.npy".format(args.data, args.dataset))
            vocab = json.load(open('{}/vocab.json'.format(args.data), 'r'))
            token_tag_map = json.load(open('{}/token_labels.json'.format(args.data), 'r'))
            index2word = dict([(v,k) for k,v in vocab.items()])
            index2tag = dict([(v, k) for k, v in token_tag_map.items()])
            with open('{}/{}.txt'.format(args.data, args.dataset), 'w') as d:
                for ins,tags in zip(X,Y):
                    for tok,tag in zip(ins,tags):
                        if index2word[tok] != '<pad>':
                            d.write(index2word[tok] + ' ' + index2tag[tag] + '\n')
                    d.write('\n')
    else:
        data = [i for i in glob('{}/*.txt'.format(args.data)) if i.__contains__('train') or i.__contains__('dev')]
        print(data)
        all_sentences, vocab, index2word, num_words, tag_map, index2tag = create_vocabularly(data_input=data,
                                                                                            output_dir=args.outputdir,
                                                                                            tokenizer=args.tokenizer)
        with open(args.vocab, 'r') as voc, open(args.token_vocab, 'r') as tok_voc:
            vocab = json.load(voc)
            tok_vocab = json.load(tok_voc)
            #pre-process data, args.method == 'LSAN'/'LWAN'
            document_matrix, sentence_label_matrix, token_label_matrix = pre_process(documents=all_sentences,
                                                                                     vocab=vocab,
                                                                                     token_label_vocab=tok_vocab,
                                                                                     method=args.method,
                                                                                     output_dir=args.outputdir)

        #utils.create_dataset_to_encode(datafiles=data, tokenizer=args.tokenizer, output_dir=args.outputdir)

