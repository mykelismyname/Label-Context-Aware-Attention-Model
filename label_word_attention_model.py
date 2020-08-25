import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, drop_out, batch_size, emb_dim, lstm_hdim, d_a, token_tag_map, sentence_tag_map, token_label_embeddings, embeddings):
        super(LabelWordAttention, self).__init__()
        self.token_classes = len(token_tag_map)
        self.sentence_classes = len(sentence_tag_map)
        self.embeddings = self.load_embeddings(embeddings)
        self.token_label_embeddings = self.load_embeddings(token_label_embeddings)

        self.word_lstm = nn.LSTM(emb_dim, hidden_size=lstm_hdim, num_layers=1, batch_first=True, bidirectional=True)
        self.linear_first = nn.Linear(lstm_hdim*2, d_a)
        self.linear_second = nn.Linear(d_a, len(token_tag_map))

        self.weight1 = nn.Linear(lstm_hdim * 2, 1)
        self.weight2 = nn.Linear(lstm_hdim * 2, 1)

        self.output_layer = nn.Linear(lstm_hdim * 2, len(token_tag_map))
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hdim

    def init_hidden(self):
        return (torch.randn(2, 1, self.lstm_hid_dim).cuda(),
                torch.randn(2, 1, self.lstm_hid_dim).cuda())

    def load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, x, abstract_encoder=False):
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        embeddings = embeddings.expand(1, embeddings.size(0), embeddings.size(1))
        # print('Embedding of a token ', embeddings.size())
        # step1 get LSTM outputs
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.word_lstm(embeddings, hidden_state)
        # print('LSTM ouput: ', outputs.shape, 'LSTM hidden state: ', hidden_state[0].shape)
        # step2 get self-attention
        selfatt = torch.tanh(self.linear_first(outputs))
        selfatt = self.linear_second(selfatt)
        selfatt = F.softmax(selfatt, dim=1)
        # print('Self Attention: ', selfatt.shape)
        selfatt = selfatt.transpose(1, 2)
        self_att = torch.bmm(selfatt, outputs)
        # print('Self attention token representation: ', self_att.shape)
        # step3 get label-attention
        h1 = outputs[:, :, :self.lstm_hid_dim]
        h2 = outputs[:, :, self.lstm_hid_dim:]

        token_label_embed = self.token_label_embeddings.weight.data
        # print('token label matrix size: ', token_label_embed.shape, 'h1: ', h1.shape)
        m1 = torch.bmm(token_label_embed.expand(1, self.token_classes, self.lstm_hid_dim), h1.transpose(1, 2))
        m2 = torch.bmm(token_label_embed.expand(1, self.token_classes, self.lstm_hid_dim), h2.transpose(1, 2))
        # print('M1', m1.shape, 'M2', m2.shape)
        label_att = torch.cat((torch.bmm(m1, h1), torch.bmm(m2, h2)), 2)
        # print('Label att', label_att.shape)
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
        # avg_embeddings = self.output_layer(avg_embeddings)
        pred = F.softmax(avg_embeddings, dim=1)
        return tok, pred


class AbstractEncoder(BasicModule):
    def __init__(self, drop_out, emb_dim, lstm_hdim, embeddings):
        super(AbstractEncoder, self).__init__()
        self.embeddings = self.load_embeddings(embeddings)
        self.word_lstm = nn.LSTM(emb_dim, hidden_size=lstm_hdim, num_layers=1, batch_first=True, bidirectional=True)
        self.embedding_dropout = nn.Dropout(p=drop_out)
        self.abstract_output = nn.Linear(2*lstm_hdim, lstm_hdim)
        self.lstm_hid_dim = lstm_hdim

    def init_hidden(self):
        return (torch.randn(2, 1, self.lstm_hid_dim).cuda(),
                torch.randn(2, 1, self.lstm_hid_dim).cuda())

    def load_embeddings(self, embeddings):
        """Load the embeddings based on flag"""
        embeddings_ = torch.nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        embeddings_.weight = torch.nn.Parameter(embeddings)
        return embeddings_

    def forward(self, x, abstract_encoder=False):
        embeddings = self.embeddings(x)
        embeddings = self.embedding_dropout(embeddings)
        embeddings = embeddings.expand(1, embeddings.size(0), embeddings.size(1))
        # step1 get LSTM outputs
        hidden_state = self.init_hidden()
        outputs, hidden_state = self.word_lstm(embeddings, hidden_state)
        outputs = torch.sum(self.abstract_output(outputs), dim=1)
        return outputs

class LabelSentenceAttention(BasicModule):
    def __init__(self, lstm_hdim, sentence_tag_map):
        super(LabelSentenceAttention, self).__init__()
        self.sent_classes = len(sentence_tag_map)
        self.output_layer = nn.Linear(lstm_hdim * 2, len(sentence_tag_map))

    def forward(self, sentence):
        sentence = sentence.expand(1, sentence.size(0))
        sentence = self.output_layer(sentence)
        pred = torch.sigmoid(sentence)
        return pred

