import os
import json
import numpy as np
from tqdm import tqdm
from mxnet.contrib import text
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
import torch
import torch.utils.data as data_utils
import label_word_attention_model as lwan
import argparse
import utils as ut
from pprint import pprint
from colored import fg, bg, attr
import chk_pnts
import time
from datetime import datetime

def train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, outputdir, vocab_map,
          token_tag_map, sentence_tag_map, sent_pool='mean', epochs=5, abs_encoder=True, GPU=True):
    if GPU:
        lwan_model.cuda()
        abs_encoder_model.cuda()
        san_model.cuda()
    min_loss = np.inf
    results_file = open(os.path.join(outputdir, 'results.txt'), 'w')
    vocab_map = dict([(v, k) for k, v in vocab_map.items()])
    start_time = time.time()
    results_file.write('Start time in seconds: {}'.format(start_time))
    for i in range(epochs):
        print("Running EPOCH", i + 1)
        results_file.write('EPOCH : {}'.format(i + 1) + '\n')
        prec_k, ndcg_k = [], []
        train_loss = []
        t_preds, t_true, s_preds, s_true = [], [], [], []
        sent_toks = []
        for batch_idx, train in enumerate(tqdm(train_loader)):
            sentences_preds = torch.empty(0).cuda()
            token_preds = torch.empty(0).cuda()
            sentence_labels = train[2]
            token_labels = train[1]
            combined_loss = []
            for sent, tags, label in zip(train[0], token_labels, sentence_labels):
                opt.zero_grad()
                tags = tags.unsqueeze(-1).cuda()
                sent = np.array([i for i in sent if int(i) != 0])  # remove padding
                s_toks = [vocab_map[i] for i in sent]
                sent = torch.from_numpy(sent)
                sent = sent.unsqueeze(-1).cuda()
                abstract_encoded = abs_encoder_model(sent).cuda()
                tokens, y_pred = lwan_model(sent, abstract_encoded, abstract_encoder=abs_encoder)
                sent_toks.append(s_toks)
                tags = oneHotEncoder(tags, 21)
                token_preds = torch.cat((token_preds, y_pred), dim=0)
                seq_loss = criterion(y_pred, tags[:y_pred.shape[0], :].cuda().float())

                for j in [b.tolist() for b in tags[:y_pred.shape[0], :]]:
                    t_true.append(j.index(1))

                t_topv, t_topi = y_pred.topk(1)
                for h in t_topi:
                    t_preds.append(h.item())

                # sentence level
                if sent_pool == 'mean':
                    sentence = torch.mean(tokens, dim=1)
                elif sent_pool == 'max':
                    sentence = torch.max(tokens, dim=1)
                else:
                    sentence = torch.sum(tokens, dim=1)
                sentence = torch.add(sentence, abstract_encoded.squeeze(1)) if abs_encoder else sentence
                sentence = sentence.unsqueeze(0)
                sentence_pred = san_model(sentence, sent_pool, abstract_encoded, abstract_encoder=abs_encoder)
                label = label.expand(1, label.shape[0])

                for m in [n.tolist() for n in label]:
                    s_true.append(m)

                s_topv, s_topi = sentence_pred.topk(1)
                for k in sentence_pred:
                    s_preds.append([i.item() for i in k])

                sent_loss = criterion(sentence_pred, label.cuda().float())
                sentences_preds = torch.cat((sentences_preds, sentence_pred), dim=0)
                loss = seq_loss + sent_loss
                loss.backward()
                opt.step()
                combined_loss.append(float(loss))

            labels_cpu = sentence_labels.data.cpu().float()
            pred_cpu = sentences_preds.data.cpu()
            prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            prec_k.append(prec)
            ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
            ndcg_k.append(ndcg)
            loss_ = np.mean(combined_loss)
            train_loss.append(loss_)

        avg_loss = np.mean(train_loss)
        epoch_prec = np.array(prec_k).mean(axis=0)
        epoch_ndcg = np.array(ndcg_k).mean(axis=0)
        t_classification = f_measure(target=t_true, predicted=t_preds, sent_toks=sent_toks, token_tag_map=token_tag_map,
                                     sentence_tag_map=sentence_tag_map, epoch=i + 1, outputdir=outputdir, level='token')
        s_classification = f_measure(target=s_true, predicted=s_preds, sent_toks=sent_toks, token_tag_map=token_tag_map,
                                     sentence_tag_map=sentence_tag_map, epoch=i + 1, outputdir=outputdir,
                                     level='sentence')
        print(t_classification)
        print(s_classification)
        print("epoch %2d train end : avg_loss = %.4f" % (i + 1, avg_loss))
        print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        epoch_prec[0], epoch_prec[2], epoch_prec[4]))
        print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]))
        results_file.write('Token level\n{}\nSentence level\n{}'.format(t_classification, s_classification))
        results_file.write("epoch {} train end : avg_loss = {:.4f}".format(i + 1, avg_loss) + '\n')
        results_file.write("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
        epoch_prec[0], epoch_prec[2], epoch_prec[4]) + '\n')
        results_file.write(
            "ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (epoch_ndcg[0], epoch_ndcg[2], epoch_ndcg[4]) + '\n\n')

        # eva;uation
        results_file.write('Evaluation\n')
        print('\n+++++++++++++++++++++EVALUATION++++++++++++++++++++++++\n')
        joint_loss = evaluate(epoch=i + 1,
                             test_loader=test_loader,
                             lwan_model=lwan_model,
                             abs_encoder_model=abs_encoder_model,
                             san_model=san_model,
                             sent_pool=sent_pool,
                             results_file=results_file, output_dir=outputdir, vocab_map=vocab_map, token_tag_map=token_tag_map,
                             sentence_tag_map=sentence_tag_map, criterion=criterion, abs_encoder=True)
    end_time = time.time()
    results_file.write('End time in seconds: {}\nTotal duration in minutes: {}'.format(end_time, float(
        (end_time - start_time) / 60)))

        # # save checkpoint
        # # create checkpoint variable and add important data
        # checkpoint = {
        #     'epoch': i+1,
        #     'valid_loss_min':joint_loss,
        #     'state_dict': model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        # }
        # chk_pnts.save_model_check_point(checkpoint, False, checkpoint_path, best_model_path)
        #
        # ## TODO: save the model if validation loss has decreased
        # if joint_loss <= min_loss:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, joint_loss))
        #     # save checkpoint as best model
        #     chk_pnts.save_model_check_point(checkpoint, True, checkpoint_path, best_model_path)
        #     valid_loss_min = valid_loss

def evaluate(epoch, test_loader, lwan_model, abs_encoder_model, san_model, sent_pool, results_file, output_dir,
             vocab_map, token_tag_map, sentence_tag_map, criterion, abs_encoder=True):
    prec_k, ndcg_k = [], []
    test_loss = []
    t_preds, t_true, s_preds, s_true = [], [], [], []
    sent_toks = []
    for batch_idx, test in enumerate(tqdm(test_loader)):
        sentences_preds = torch.empty([0]).cuda()
        sentence_labels = test[2]
        token_labels = test[1]
        batch_loss = []
        for sent, tags, label in zip(test[0], token_labels, sentence_labels):
            s_toks = [vocab_map[i.item()] for i in sent]
            tags = tags.unsqueeze(-1).cuda()
            sent = np.array([i for i in sent if int(i) != 0])  # remove padding
            s_toks = [vocab_map[i] for i in sent]
            sent = torch.from_numpy(sent)
            sent = sent.unsqueeze(-1).cuda()
            abstract_encoded = abs_encoder_model(sent).cuda()
            tokens, y_pred = lwan_model(sent, abstract_encoded, abstract_encoder=abs_encoder)
            sent_toks.append(s_toks)
            tags = oneHotEncoder(tags, 21)
            seq_loss = criterion(y_pred, tags[:y_pred.shape[0], :].cuda().float())

            for j in [b.tolist() for b in tags[:y_pred.shape[0], :]]:
                t_true.append(j.index(1))

            t_topv, t_topi = y_pred.topk(1)
            for h in t_topi:
                t_preds.append(h.item())

            # sentence level
            if sent_pool == 'mean':
                sentence = torch.mean(tokens, dim=1)
            elif sent_pool == 'max':
                sentence = torch.max(tokens, dim=1)
            else:
                sentence = torch.sum(tokens, dim=1)

            sentence = torch.add(sentence, abstract_encoded.squeeze(1)) if abs_encoder else sentence
            sentence = sentence.unsqueeze(0)
            sentence_pred = san_model(sentence, sent_pool, abstract_encoded, abstract_encoder=abs_encoder)
            label = label.expand(1, label.shape[0])

            for m in [n.tolist() for n in label]:
                s_true.append(m)

            s_topv, s_topi = sentence_pred.topk(1)
            for k in sentence_pred:
                s_preds.append([i.item() for i in k])

            sentences_preds = torch.cat((sentences_preds, sentence_pred), dim=0)
            sent_loss = criterion(sentence_pred, label.cuda().float())
            loss = seq_loss + sent_loss
            batch_loss.append(float(loss))

        test_loss.append(np.mean(batch_loss))
        labels_cpu = sentence_labels.data.cpu().float()
        pred_cpu = sentences_preds.data.cpu()
        prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
        prec_k.append(prec)
        ndcg = Ndcg_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
        ndcg_k.append(ndcg)

    avg_test_loss = np.mean(test_loss)
    test_prec = np.array(prec_k).mean(axis=0)
    test_ndcg = np.array(ndcg_k).mean(axis=0)
    t_classification = f_measure(target=t_true, predicted=t_preds, sent_toks=sent_toks, token_tag_map=token_tag_map,
                                 sentence_tag_map=sentence_tag_map, epoch=epoch, outputdir=output_dir, level='token',
                                 evaluation=True)
    s_classification = f_measure(target=s_true, predicted=s_preds, sent_toks=sent_toks, token_tag_map=token_tag_map,
                                 sentence_tag_map=sentence_tag_map, epoch=epoch, outputdir=output_dir, level='sentence',
                                 evaluation=True)
    print(t_classification)
    print(s_classification)
    print("epoch %2d test end : avg_loss = %.4f" % (epoch, avg_test_loss))
    print("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (test_prec[0], test_prec[2], test_prec[4]))
    print("ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]))

    results_file.write('Token level\n{}\nSentence level\n{}'.format(t_classification, s_classification))
    results_file.write("epoch {} train end : avg_loss = {:.4f}".format(epoch, avg_test_loss) + '\n')
    results_file.write("precision@1 : %.4f , precision@3 : %.4f , precision@5 : %.4f " % (
    test_prec[0], test_prec[2], test_prec[4]) + '\n')
    results_file.write(
        "ndcg@1 : %.4f , ndcg@3 : %.4f , ndcg@5 : %.4f " % (test_ndcg[0], test_ndcg[2], test_ndcg[4]) + '\n\n')

    return avg_test_loss


def f_measure(target, predicted, sent_toks, token_tag_map, sentence_tag_map, epoch, outputdir, level, evaluation=False, prfs=False):
    if level == 'token':
        cls_reversed = dict([(v, k) for k, v in token_tag_map.items()])
        Y_t = [cls_reversed[i] for i in target]
        Y_p = [cls_reversed[i] for i in predicted]
        cls = list(cls_reversed.values())
        if evaluation and epoch > 8:
            if len(Y_t) == len(Y_p):
                k = [i for j in sent_toks for i in j]
                print(1111, '\n', len(Y_t), len(Y_p), len(k), len(sent_toks), '\n')
                write_out_output(Y_t, Y_p, epoch, sent_toks, outputdir)
            else:
                raise KeyError('Some either missing or extra tokens')
    elif level == 'sentence':
        cls_reversed = sentence_tag_map
        target = np.array(target)
        predicted = np.array(predicted)
        p_mat = predicted >= 0.5
        p_mat = p_mat.astype(int)

        Y_t, Y_p, o = [], [], 1
        x = open(os.path.join(outputdir, 'sentence_predictions.txt'), 'a')
        for yt, yp in zip(target, p_mat):
            yt = binary_to_labels(val=yt.tolist(), labels=cls_reversed)
            yp = binary_to_labels(val=yp.tolist(), labels=cls_reversed)
            if evaluation:
                x.write('{}. {} {}'.format(o, yp, yt))
                x.write('\n')
                o += 1
            for m, n in zip(yt, yp):
                Y_t.append(m)
                Y_p.append(n)
        x.write('\n')
        cls = list(cls_reversed.values())
        cls.append(0)

    classification = classification_report(Y_t, Y_p, labels=cls)

    if prfs:
        F_measure = f1_score(target, predicted, average='macro')
        recall = recall_score(target, predicted, average=None)
        precision = precision_score(target, predicted, average=None)
        AP = precision_score(target, predicted, average='macro')
        fig, pr = plt.subplots()
        pr.plot(recall, precision, marker='+')
        pr.set_title('Precision-recall Curve')
        plt.legend()
        return F_measure, AP, fig, classification
    return classification


def viz_text():
    text_col_map = {
        "B-Adverse-effects": bg('blue'),
        "B-Life-Impact": bg('red'),
        "B-Mortality": bg('dark_gray'),
        "B-Physiological-Clinical": bg('green'),
        "B-Resource-use": bg('cyan'),
        "B-outcome": bg('yellow'),
        "I-Adverse-effects": bg('blue'),
        "I-Life-Impact": bg('red'),
        "I-Mortality": bg('dark_gray'),
        "I-Physiological-Clinical": bg('green'),
        "I-Resource-use": bg('cyan'),
        "I-outcome": bg('yellow'),
        "O": fg('black')
    }
    return text_col_map


def write_out_output(Y_t, Y_p, epoch, sent_toks, outputdir):
    a = b = 0
    col_map = viz_text()
    with open(outputdir + '/epoch_{}_predictions.txt'.format(epoch), 'w') as pre:
        for x, _ in enumerate(Y_t):
            if a == x:
                print_out = ''
                for tok in sent_toks[b]:
                    pre.write(tok + ' ' + Y_t[a] + ' ' + Y_p[a] + '\n')
                    if Y_p[a] == "O":
                        print_out += '{} {} {}'.format(col_map[Y_p[a]], tok, attr('reset'))
                    elif Y_p[a].__contains__("-"):
                        print_out += '{}{}{} {} {}'.format(fg('white'), col_map[Y_p[a]], attr('bold'), tok,
                                                           attr('reset'))
                    elif Y_p[a] != "O":
                        print_out += '{}{} {} {}'.format(fg('dark_gray'), attr('bold'), tok, attr('reset'))
                    else:
                        print('\n', tok, Y_p[a], '\n')

                    a += 1
                b += 1
                print_out += '\n\n'
                pre.write('\n')
                print(b, print_out.strip())
        pre.close()


def oneHotEncoder(x, label_size):
    x_ = torch.zeros((x.shape[0], label_size))
    for i in range(len(x)):
        x_[i][x[i].item()] = 1
    return x_


def binary_to_labels(val, labels):
    ind = [i for i, j in enumerate(val) if j == 1]
    for i in ind:
        val[i] = labels[i]
    return val


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


def get_factor(label_count, k):
    res = []
    for i in range(len(label_count)):
        n = int(min(label_count[i], k))
        f = 0.0
        for j in range(1, n + 1):
            f += 1 / np.log(j + 1)
        res.append(f)
    return np.array(res)


def main(args):
    X_train = np.load("{}/X_train.npy".format(args.data))
    X_label_train = np.load("{}/X_label_train.npy".format(args.data))
    Y_train = np.load("{}/Y_train.npy".format(args.data))
    X_dev = np.load("{}/X_dev.npy".format(args.fixed_test if args.fixed_test else args.data))
    X_label_dev = np.load("{}/X_label_dev.npy".format(args.fixed_test if args.fixed_test else args.data))
    Y_dev = np.load("{}/Y_dev.npy".format(args.fixed_test if args.fixed_test else args.data))
    token_label_embed = np.load("{}/token_labels_embed.npy".format(args.vocabularly))
    sentence_label_embed = np.load("{}/sentence_labels_embed.npy".format(args.vocabularly))
    word_embed = text.embedding.CustomEmbedding('{}/word_embed.txt'.format(args.vocabularly))
    token_tag_map = json.load(open('{}/token_labels.json'.format(args.vocabularly), 'r'))
    vocab_map = json.load(open('{}/vocab.json'.format(args.vocabularly), 'r'))
    current_date_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    outputdir = ut.create_directories_per_series_des(args.data + '/LCAM-Bilstm{}{}_{}'.format(args.model_name, args.model_name2, current_date_time))
    sentence_tags = [i.strip() for i in open('{}/sentence_labels.txt'.format(args.vocabularly), 'r').readlines()]
    sentence_tag_map = dict([(k, v) for k, v in enumerate(sentence_tags)])

    train_data = data_utils.TensorDataset(torch.from_numpy(X_train).type(torch.LongTensor),
                                          torch.from_numpy(X_label_train).type(torch.LongTensor),
                                          torch.from_numpy(Y_train).type(torch.LongTensor))

    dev_data = data_utils.TensorDataset(torch.from_numpy(X_dev).type(torch.LongTensor),
                                        torch.from_numpy(X_label_dev).type(torch.LongTensor),
                                        torch.from_numpy(Y_dev).type(torch.LongTensor))

    train_loader = data_utils.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
    test_loader = data_utils.DataLoader(dev_data, batch_size=64, drop_last=True)

    word_embed = word_embed.idx_to_vec.asnumpy()
    word_embed = torch.from_numpy(word_embed).float()
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
                                            drop_out=0.3,
                                            d_a=200,
                                            sentence_tag_map=sentence_tag_map,
                                            sentence_label_embeddings=sentence_label_embed)

    abs_encoder_model = lwan.AbstractEncoder(drop_out=0.3,
                                             emb_dim=300,
                                             lstm_hdim=300,
                                             embeddings=word_embed)

    criterion = torch.nn.BCELoss()
    combined_params = list(lwan_model.parameters()) + list(san_model.parameters())
    opt = torch.optim.Adam(combined_params, lr=0.001)
    train(lwan_model, abs_encoder_model, san_model, train_loader, test_loader, criterion, opt, outputdir, vocab_map,
          token_tag_map, sentence_tag_map, sent_pool='mean',
          epochs=10, abs_encoder=True, GPU=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data source')
    parser.add_argument('--fixed_test', type=str, help='data source')
    parser.add_argument('--vocabularly', type=str, required=True, help='directory with vocabularly and all other LCAM token and sentence embeddings')
    parser.add_argument('--transformer_model', type=str, help='transformer model BioBERT/ALBERT')
    parser.add_argument('--abs_encoder', action='store_true', help='Include context from the abstract or not')
    parser.add_argument('--attention', action='store_true', help='Include or eliminate attention')
    args = parser.parse_args()
    args.model_name = '+Abstract' if args.abs_encoder else '-Abstract'
    args.model_name2 = '+Attention' if args.attention else '-Attention'
    main(args)