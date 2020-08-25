import os
import json
import numpy as np
from tqdm import tqdm
from mxnet.contrib import text
import torch
import torch.utils.data as data_utils
import label_word_attention_model as lwan

def train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, sent_pool='sum', epochs = 5, GPU=True):
    if GPU:
        lwan_model.cuda()
        abs_encoder_model.cuda()
        san_model.cuda()
    for i in range(epochs):
        print("Running EPOCH",i+1)
        prec_k = []
        ndcg_k = []
        train_loss = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            sentences = torch.empty(0).cuda()
            labels = train[2]
            batch_loss = []
            for sent,tags,label in zip(train[0], train[1], labels):
                opt.zero_grad()
                # obtain the abstract representation print('sent', sent.size())
                label = label.expand(1, label.size(0))
                abstract_encoded = abs_encoder_model(sent.cuda())
                sentence = torch.empty(0).cuda()
                token_seq_loss = torch.empty(0).cuda()
                print(sent)
                break
            break
        #         # print(sent.shape, tags.shape)
        #         for token,tag in zip(sent, tags):
        #             x, y = token.cuda(), tag.cuda()
        #             x, y_ = x.expand(1), y.expand(1)
        #             tok, y_pred= lwan_model(x)
        #             tok = torch.sum(tok, 1)
        #             y_ = oneHotEncoder(y_).cuda()
        #             seq_loss = criterion(y_pred, y_)
        #             token_seq_loss = torch.cat((token_seq_loss, seq_loss.unsqueeze(0)))
        #             sentence = torch.cat((sentence, tok), dim=0)
        #
        #         #aggregating token encodings to obtain a sentence vector
        #         if sent_pool == 'mean':
        #             sentence = torch.mean(sentence, dim=0)
        #         elif sent_pool == 'max':
        #             sentence = torch.max(sentence, dim=0)
        #         else:
        #             sentence = torch.sum(sentence, dim=0)
        #
        #         print(sentence.shape)
        #         sentence_pred = san_model(sentence)
        #         sent_loss = criterion(sentence_pred, label.cuda().float())
        #         loss = torch.mean(token_seq_loss) + sent_loss
        #         loss.backward()
        #         print('token loss: {} and sent loss {}'.format(seq_loss, sent_loss))
        #         opt.step()
        #         batch_loss.append(float(loss))
        #         sentences = torch.cat((sentences, sentence_pred), dim=0)
        #
        #     print(labels.shape, sentence.shape)
        #     train_loss.append(np.mean(batch_loss))
        #     sentences = sentences.detach().cpu()
        #     labels = labels.cpu().float()
        #     prec = precision_k(labels.numpy(), sentences.numpy(), 5)
        #     prec_k.append(prec)
        #     ndcg = Ndcg_k(labels.numpy(), sentences.numpy(), 5)
        #     ndcg_k.append(ndcg)
        #
        # avg_loss = np.mean(train_loss)
        # epoch_prec = np.array(prec_k).mean(axis=0)
        # epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        # print("epoch {} train end : avg_loss = {:.4f}".format(i+1, avg_loss))
        # print(epoch_prec, epoch_prec, epoch_prec)
        # print("precision@1 : {:.4f} , precision@3 : {:.4f} , precision@5 : {:.4f} ".format(epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        # print("ndcg@1 : {:.4f} , ndcg@3 : {:.4f} , ndcg@5 : {:.4f} ".format(epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        #
        # # eva;uation
        # avg_test_loss, test_prec, test_ndcg = evaluate(test_loader=test_loader, lwan_model=lwan_model, san_model=san_model, sent_pool=sent_pool)
        # print("epoch {} test end : avg_loss = {:.4f}".format(i + 1, avg_test_loss))
        # print("precision@1 : {:.4f} , precision@3 : {:.4f} , precision@5 : {:.4f} ".format(test_prec[0], test_prec[2], test_prec[4]))
        # print("ndcg@1 : {:.4f} , ndcg@3 : {:.4f} , ndcg@5 : {:.4f} ".format(test_ndcg[0], test_ndcg[2], test_ndcg[4]))


def evaluate(test_loader, lwan_model, san_model, sent_pool):
    prec_k = []
    ndcg_k = []
    test_loss = []
    for batch_idx, test in enumerate(tqdm(test_loader)):
        sentences = torch.empty(0).cuda()
        labels = test[2]
        batch_loss = []
        for sent,tags,label in zip(test[0], test[1], test[2]):
            # obtain the abstract representation print('sent', sent.size())
            label = label.expand(1, label.size(0))
            abstract_encoded = abs_encoder_model(sent.cuda())
            sentence = torch.empty(0).cuda()
            test_token_seq_loss = []
            for token,tag in zip(sent, tags):
                x, y = token.cuda(), tag.cuda()
                x, y_ = x.expand(1), y.expand(1)
                tok, y_pred = lwan_model(x)
                tok = torch.sum(tok, 1)
                y_ = oneHotEncoder(y_).cuda()
                seq_loss = criterion(y_pred, y_)
                test_token_seq_loss.append(float(seq_loss))
                sentence = torch.cat((sentence, tok), dim=0)

            # aggregating token encodings to obtain a sentence vector
            if sent_pool == 'mean':
                sentence = torch.mean(sentence, dim=0)
            elif sent_pool == 'max':
                sentence = torch.max(sentence, dim=0)
            else:
                sentence = torch.sum(sentence, dim=0)

            sentence_pred = san_model(sentence)
            sent_loss = criterion(sentence_pred, label.cuda().float())
            loss = np.mean(test_token_seq_loss) + float(sent_loss)
            batch_loss.append(loss)
            sentences = torch.cat((sentences, sentence_pred), dim=0)

        test_loss.append(np.mean(batch_loss))
        sentences = sentences.detach().cpu()
        labels = labels.cpu().float()
        prec = precision_k(labels.numpy(), sentences.numpy(), 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels.numpy(), sentences.numpy(), 5)
        ndcg_k.append(ndcg)

    avg_test_loss = np.mean(test_loss)
    test_prec = np.array(prec_k).mean(axis=0)
    test_ndcg = np.array(ndcg_k).mean(axis=0)

    return avg_test_loss, test_prec, test_ndcg
    print("epoch %2d test end : avg_loss = %.4f" % (i+1, avg_test_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
    test_prec[0], test_prec[2], test_prec[4]))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))

def oneHotEncoder(x):
    vec = torch.zeros(22) #21 is the number of toke labels
    vec[x.item()] = 1
    return vec


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        print(score_mat, true_mat)
        mat = np.multiply(score_mat, true_mat)
        #print("mat",mat)
        num = np.sum(mat, axis=1)
        p[k] = np.mean(num / (k + 1))
    return np.around(p, decimals=4)

def Ndcg_k(true_mat, score_mat, k):
    res = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    label_count = np.sum(true_mat, axis=1)

    for m in range(k):
        y_mat = np.copy(true_mat)
        for i in range(rank_mat.shape[0]):
            y_mat[i][rank_mat[i, :-(m + 1)]] = 0
            for j in range(m + 1):
                y_mat[i][rank_mat[i, -(j + 1)]] /= np.log(j + 1 + 1)

        dcg = np.sum(y_mat, axis=1)
        factor = get_factor(label_count, m + 1)
        ndcg = np.mean(dcg / factor)
        res[m] = ndcg
    return np.around(res, decimals=4)

def get_factor(label_count,k):
    res=[]
    for i in range(len(label_count)):
        n=int(min(label_count[i],k))
        f=0.0
        for j in range(1,n+1):
            f+=1/np.log(j+1)
        res.append(f)
    return np.array(res)

X_train = np.load("multi-labelled-data/LWAN/X_train.npy")
X_label_train = np.load("multi-labelled-data/LWAN/X_label_train.npy")
Y_train = np.load("multi-labelled-data/LWAN/Y_train.npy")
X_dev = np.load("multi-labelled-data/LWAN/X_dev.npy")
X_label_dev = np.load("multi-labelled-data/LWAN/X_label_dev.npy")
Y_dev = np.load("multi-labelled-data/LWAN/Y_dev.npy")
token_label_embed = np.load("multi-labelled-data/LWAN/token_labels_embed.npy")
sentence_label_embed = np.load("multi-labelled-data/LWAN/sentence_labels_embed.npy")
word_embed = text.embedding.CustomEmbedding('multi-labelled-data/LSAN/word_embed.txt')
token_tag_map = json.load(open('multi-labelled-data/LWAN/token_labels.json', 'r'))

sentence_tag_map = [i.strip() for i in open('multi-labelled-data/LWAN/sentence_labels.txt', 'r').readlines()]

train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                      torch.from_numpy(X_label_train).type(torch.LongTensor),
                                      torch.from_numpy(Y_train).type(torch.LongTensor))

dev_data = data_utils.TensorDataset(torch.from_numpy(X_dev).type(torch.LongTensor),
                                    torch.from_numpy(X_label_dev).type(torch.LongTensor),
                                    torch.from_numpy(Y_dev).type(torch.LongTensor))

train_loader = data_utils.DataLoader(train_data, batch_size=64, drop_last=True)
test_loader = data_utils.DataLoader(dev_data, batch_size=64, drop_last=True)

word_embed = word_embed.idx_to_vec.asnumpy()
word_embed  = torch.from_numpy(word_embed).float()
token_label_embed = torch.from_numpy(token_label_embed).float()
sentence_label_embed = torch.from_numpy(sentence_label_embed).float()

lwan_model = lwan.LabelWordAttention(drop_out=0.3,
                  batch_size=64,
                  emb_dim=300,
                  lstm_hdim=300,
                  d_a=200,
                  token_tag_map=token_tag_map,
                  sentence_tag_map=sentence_tag_map,
                  token_label_embeddings=token_label_embed,
                  embeddings=word_embed)

san_model = lwan.LabelSentenceAttention(lstm_hdim=300,
                                        sentence_tag_map=sentence_tag_map)

abs_encoder_model = lwan.AbstractEncoder(drop_out=0.3,
                                        emb_dim=300,
                                        lstm_hdim=300,
                                        embeddings=word_embed)

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(lwan_model.parameters(), lr=0.001, betas=(0.9, 0.99))
train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, epochs = 5, GPU=True)