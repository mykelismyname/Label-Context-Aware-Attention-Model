import torch
import torch.nn as nn
import torch.nn.functional as F
from flair.data import Sentence
from flair.embeddings import TransformerDocumentEmbeddings, TransformerWordEmbeddings

class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            raise ValueError('Please specify the saving road!!!')
        torch.save(self.state_dict(), path)
        return path

class LabelWordAttention(BasicModule):
    def __init__(self, drop_out, batch_size, emb_dim, lstm_hdim, d_a, vocab, token_tag_map, sentence_tag_map, token_label_embeddings, transformer_embeddings):
        super(LabelWordAttention, self).__init__()
        self.token_classes = len(token_tag_map)
        self.sentence_classes = len(sentence_tag_map)
        self.word_embeddings = transformer_embeddings
        self.token_label_embeddings = token_label_embeddings
        self.vocab = vocab
        self.emb_dim = emb_dim

        self.word_lstm = nn.LSTM(emb_dim, hidden_size=lstm_hdim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_first = nn.Linear(lstm_hdim*2, d_a)
        self.linear_second = nn.Linear(d_a, len(token_tag_map))

        self.weight1 = nn.Linear(lstm_hdim * 2, 1)
        self.weight2 = nn.Linear(lstm_hdim * 2, 1)

        self.output_layer = nn.Linear(lstm_hdim * 2, len(token_tag_map))
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hdim

    def init_hidden(self, sent_len):
        return (torch.randn(2, sent_len, self.lstm_hid_dim).cuda(),
                torch.randn(2, sent_len, self.lstm_hid_dim).cuda())

    def load_token_embeddings(self, sentence, transformer_embeddings, vocab, token_label_embs=True):
        """Load the embeddings based on flag"""
        if not token_label_embs:
            sentence = [i.item() for i in sentence]
            vocab_reversed = dict([(v,k) for k,v in vocab.items()])
            sentence = [vocab_reversed[i] for i in sentence if i > 1]
            sentence = Sentence(' '.join(sentence))
            transformer_embeddings.embed(sentence)
            embeddings_ = torch.zeros((len(sentence), self.emb_dim))
            for i,tok in enumerate(sentence):
                print(tok.text)
                embeddings_[i] = tok.embedding
            embeddings_ = embeddings_.unsqueeze(1).cuda()
        else:
            embeddings = self.token_label_embeddings
            embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
            embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, x, abstract, abstract_encoder=False):
        embeddings = self.load_token_embeddings(x, self.word_embeddings, self.vocab, token_label_embs=False)
        sent_len = embeddings.size(0)
        print('Embedding of a token ', embeddings.size())
        # step1 get LSTM outputs
        hidden_state = self.init_hidden(sent_len)
        outputs, hidden_state = self.word_lstm(embeddings, hidden_state)
        print('LSTM ouput: ', outputs.shape, 'LSTM hidden state: ', hidden_state[0].shape)
        #outputs = torch.add(outputs, abstract) if abstract_encoder else outputs
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        #print('Self Attention: ', selfatt.shape)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        #print('Self attention token representation: ', self_att.shape)
        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        token_label_embed = self.load_token_embeddings(x, self.word_embeddings, self.vocab, token_label_embs=True)
        token_label_embed = token_label_embed.weight.data
        #print('token label matrix size: ', token_label_embed.shape, 'h1: ', h1.shape)
        m1 = torch.bmm(token_label_embed.expand(sent_len, self.token_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(token_label_embed.expand(sent_len, self.token_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        #print('M1', m1.shape, 'M2', m2.shape)
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
        #print('Label att', label_att.shape)
        # label_att = F.normalize(label_att, p=2, dim=-1)
        # self_att = F.normalize(self_att, p=2, dim=-1) #all can
        weight1 = torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        tok = weight1 * label_att + weight2 * self_att
        #print('Final token representation: ', tok.shape)
        # there two method, for simple, just add
        # also can use linear to do it
        avg_embeddings = torch.sum(tok, 2) / self.token_classes
        #print('av embeddings', avg_embeddings.shape)
        # avg_embeddings = self.output_layer(avg_embeddings)
        # print('av embeddings', avg_embeddings.shape)
        pred = F.sigmoid(avg_embeddings)
        return tok, pred


class AbstractEncoder(BasicModule):
    def __init__(self, drop_out, emb_dim, lstm_hdim, embeddings):
        super(AbstractEncoder, self).__init__()
        self.embeddings = self.load_embeddings(embeddings)
        self.word_lstm = nn.LSTM(emb_dim, hidden_size=lstm_hdim, num_layers=1, batch_first=True, bidirectional=True)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.lstm_hid_dim = lstm_hdim

    def init_hidden(self, sent_len):
        return (torch.randn(2, sent_len, self.lstm_hid_dim).cuda(),
                torch.randn(2, sent_len, self.lstm_hid_dim).cuda())

    def load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, x, abstract_encoder=False):
        #print(x.shape)
        sent_len = len(x)
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        #print(embeddings.shape)
        #embeddings = embeddings.expand(1, embeddings.size(0), embeddings.size(1))
        # step1 get LSTM outputs
        hidden_state = self.init_hidden(sent_len)
        outputs, hidden_state = self.word_lstm(embeddings, hidden_state)
        return outputs

class LabelSentenceAttention(BasicModule):
    def __init__(self, lstm_hdim, drop_out, d_a, sentence_tag_map, sentence_label_embeddings):
        super(LabelSentenceAttention, self).__init__()
        self.sent_classes = len(sentence_tag_map)
        self.lstm_hid_dim = lstm_hdim
        self.sentence_label_embed = self.load_embeddings(sentence_label_embeddings)
        self.linear_first = nn.Linear(lstm_hdim * 2, d_a)
        self.linear_second = nn.Linear(d_a, len(sentence_tag_map))
        self.embedding_dropout = nn.Dropout(p=drop_out)

        self.weight1 = nn.Linear(lstm_hdim * 2, 1)
        self.weight2 = nn.Linear(lstm_hdim * 2, 1)
        self.output_layer = nn.Linear(lstm_hdim * 2, len(sentence_tag_map))

    def load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, sentence, sent_pool, abstract, abstract_encoder=False):
        sentence = self.embedding_dropout(sentence)
        # step1 get self-attention
        selfatt = torch.tanh(self.linear_first(sentence))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        #print('Self Attention: ', selfatt.shape)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, sentence)
        #print('Self attention token representation: ', self_att.shape)

        h1 = sentence[:, :, :self.lstm_hid_dim]
        h2 = sentence[:, :, self.lstm_hid_dim:]
        sentence_label_embed = self.sentence_label_embed.weight.data
        #print('sentence label matrix size: ', sentence_label_embed.shape)
        m1 = torch.bmm(sentence_label_embed.expand(1, self.sent_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(sentence_label_embed.expand(1, self.sent_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        #print('M1', m1.shape, 'M2', m2.shape)
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
        #print('Label att', label_att.shape)

        weight1 = torch.sigmoid(self.weight1(label_att))
        weight2 = torch.sigmoid(self.weight2(self_att))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        doc = weight1 * label_att + weight2 * self_att
       # print('Final representation', doc.shape)
        # there two method, for simple, just add
        # also can use linear to do it
        avg_sentence_embeddings = torch.sum(doc, 1) / self.sent_classes
        pred = torch.sigmoid(self.output_layer(avg_sentence_embeddings))
        #print(pred.shape)
        return pred




