import os
import json
import numpy as np
from tqdm import tqdm
from mxnet.contrib import text
import torch
import torch.utils.data as data_utils
import label_word_attention_model as lwan

def train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, sent_pool='mean', epochs = 5, GPU=True):
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
            combined_loss = []
            for sent,tags,label in zip(train[0], train[1], labels):
                opt.zero_grad()
                tags = tags.expand(1, tags.shape[0])
                tags = tags.transpose(1, 0)
                sent = sent.expand(1, sent.shape[0])
                sent = sent.transpose(1, 0).cuda()
                abstract_encoded = abs_encoder_model(sent)
                tokens, y_pred = lwan_model(sent, abstract_encoded, abstract_encoder=True)
                tags = oneHotEncoder(tags, 22)
                seq_loss = criterion(y_pred, tags.cuda().float())

                #sentence level
                sentence = torch.sum(tokens, dim=1)
                sentence_pred = san_model(sentence, sent_pool, abstract_encoded, abstract_encoder=True)
                sentences = torch.cat((sentences, sentence_pred), dim=0)
                label = label.expand(1, label.shape[0])
                #print(sentence_pred, label)
                sent_loss = criterion(sentence_pred, label.cuda().float())
                loss = seq_loss + sent_loss
                loss.backward()
                opt.step()
                combined_loss.append(float(loss))

            labels_cpu = labels.data.cpu().float()
            pred_cpu = sentences.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            loss_ = np.mean(combined_loss)
            #print(loss_)
            train_loss.append(loss_)

        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        print("epoch %2d train end : avg_loss = %.4f" % (i + 1, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))

        # eva;uation
        evaluate(epoch=i+1, test_loader=test_loader, lwan_model=lwan_model, san_model=san_model, sent_pool=sent_pool)


def evaluate(epoch, test_loader, lwan_model, san_model, sent_pool):
    prec_k = []
    ndcg_k = []
    test_loss = []
    for batch_idx, test in enumerate(tqdm(test_loader)):
        sentences = torch.empty(0).cuda()
        labels = test[2]
        batch_loss = []
        for sent,tags,label in zip(test[0], test[1], test[2]):
            tags = tags.expand(1, tags.shape[0])
            tags = tags.transpose(1, 0)
            sent = sent.expand(1, sent.shape[0])
            sent = sent.transpose(1, 0).cuda()
            abstract_encoded = abs_encoder_model(sent)
            tokens, y_pred = lwan_model(sent, abstract_encoded, abstract_encoder=True)
            tags = oneHotEncoder(tags, 22)
            seq_loss = criterion(y_pred, tags.cuda().float())

            # sentence level
            sentence = torch.sum(tokens, dim=1)
            sentence_pred = san_model(sentence, sent_pool, abstract_encoded, abstract_encoder=True)
            sentences = torch.cat((sentences, sentence_pred), dim=0)
            label = label.expand(1, label.shape[0])
            # print(sentence_pred, label)
            sent_loss = criterion(sentence_pred, label.cuda().float())
            loss = seq_loss + sent_loss
            batch_loss.append(float(loss))

        test_loss.append(np.mean(batch_loss))
        labels_cpu = labels.data.cpu().float()
        pred_cpu = sentences.data.cpu()
        prec = precision_k(labels.numpy(), pred_cpu.numpy(), 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels.numpy(), pred_cpu.numpy(), 5)
        ndcg_k.append(ndcg)

    avg_test_loss = np.mean(test_loss)
    test_prec = np.array(prec_k).mean(axis=0)
    test_ndcg = np.array(ndcg_k).mean(axis=0)

    print("epoch %2d test end : avg_loss = %.4f" % (epoch, avg_test_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
    test_prec[0], test_prec[2], test_prec[4]))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))

def oneHotEncoder(x, label_size):
    x_ = torch.zeros((x.shape[0], label_size))
    for i in range(len(x)):
        x_[i][x[i].item()] = 1
    return x_


def precision_k(true_mat, score_mat, k):
    p = np.zeros((k, 1))
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    for k in range(k):
        score_mat = np.copy(backup)
        for i in range(rank_mat.shape[0]):
            print(rank_mat[i, :-(k + 1)])
            print(score_mat[i])
            score_mat[i][rank_mat[i, :-(k + 1)]] = 0
        score_mat = np.ceil(score_mat)
        #         kk = np.argwhere(score_mat>0)
        mat = np.multiply(score_mat, true_mat)
        #         print("mat",mat)
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
combined_params = list(lwan_model.parameters()) + list(san_model.parameters())
opt = torch.optim.Adam(combined_params, lr=0.0001)
train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, epochs = 10, GPU=True)