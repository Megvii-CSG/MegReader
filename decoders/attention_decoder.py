import torch
import torch.nn as nn
import config
import numpy as np
import math

from concern.charsets import DefaultCharset


class AttentionDecoder(nn.Module):
    def __init__(self, in_channels, charset=DefaultCharset(),
                 inner_channels=512, max_size=32, height=1,
                 gt_as_output=None, step_dropout=0, **kwargs):
        super(AttentionDecoder, self).__init__()

        self.inner_channels = inner_channels
        self.encode = self._init_encoder(in_channels)

        self.max_size = max_size
        self.charset = charset
        self.height = height
        self.decoder = AttentionRNNCell(
            inner_channels, max_size + height, len(charset))
        self.step_dropout = step_dropout

        self.onehot_embedding_x = nn.Embedding(max_size, max_size)
        self.onehot_embedding_x.weight.data = torch.eye(max_size)
        self.onehot_embedding_y = nn.Embedding(height, height)
        self.onehot_embedding_y.weight.data = torch.eye(height)

        self.gt_as_output = gt_as_output
        self.loss_function = nn.NLLLoss(reduction='none')

    def _init_encoder(self, in_channels, stride=(2, 1), padding=(0, 1)):
        encode = nn.Sequential(
            self.conv_bn_relu(in_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            self.conv_bn_relu(self.inner_channels, self.inner_channels),
            nn.MaxPool2d(stride, stride, (0, 0)),
            self.conv_bn_relu(self.inner_channels, self.inner_channels,
                              kernel_size=(2, 3),
                              stride=stride, padding=padding),
        )
        return encode

    def _get_gt_as_output(self):
        if self.gt_as_output is not None:
            return self.gt_as_output
        return np.random.rand() < 0.5

    def conv_bn_relu(self, input_channels, output_channels,
                     kernel_size=3, stride=1, padding=1):
        return nn.Sequential(nn.Conv2d(input_channels, output_channels,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding),
                             nn.BatchNorm2d(output_channels),
                             nn.ReLU(inplace=True), )

    def forward(self, feature,
                targets=None,
                lengths=None,
                train=False):
        decoder_input_sequence = self.encode(feature)
        index_y, index_x = torch.meshgrid(torch.linspace(0, self.height - 1, self.height),
                                          torch.linspace(0, self.max_size - 1, self.max_size))
        index_y = index_y.to(feature.device).type(torch.long)
        index_x = index_x.to(feature.device).type(torch.long)

        batch_size = feature.shape[0]
        onehot_embedded_x = self.onehot_embedding_x(index_x).permute(2, 0, 1)
        onehot_embedded_x = onehot_embedded_x.repeat(batch_size, 1, 1, 1)
        onehot_embedded_y = self.onehot_embedding_y(index_y).permute(2, 0, 1)
        onehot_embedded_y = onehot_embedded_y.repeat(batch_size, 1, 1, 1)
        decoder_input = torch.cat(
            [decoder_input_sequence, onehot_embedded_y, onehot_embedded_x],
            dim=1)
        decoder_input = decoder_input.view(
            batch_size, decoder_input.shape[1], -1).permute(2, 0, 1)

        onehot_bos = torch.zeros(batch_size, ) + self.charset.blank
        decoder_hidden = torch.zeros(batch_size, self.inner_channels).to(feature.device)

        if self.training:
            max_steps = lengths.max()
            timestep_input = onehot_bos.to(feature.device)
            targets = targets.type(torch.long)
            for timestep in range(self.max_size):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    timestep_input, decoder_hidden, decoder_input, train=True)
                mask = (timestep <= lengths).type(torch.float)
                if timestep == 0:
                    loss = self.loss_function(
                        decoder_output, targets[:, timestep]) * mask
                    attention_pred = decoder_attention.unsqueeze(1)
                else:
                    attention_pred = torch.cat(
                        [attention_pred, decoder_attention.unsqueeze(1)], dim=1)
                    loss += self.loss_function(decoder_output,
                                               targets[:, timestep]) * mask
                i = decoder_output.argmax(dim=1)

                if self._get_gt_as_output():
                    timestep_input = targets[:, timestep]
                else:
                    timestep_input = i.detach()  # will never require gradients

                factor = (torch.rand(*timestep_input.shape) < self.step_dropout).long().to(feature.device)
                timestep_input = timestep_input.to(feature.device) * \
                                 (1 - factor) + torch.randint(high=len(self.charset),
                                                              size=timestep_input.shape).to(feature.device) * factor

                # if timestep > max_steps: break
            return loss, attention_pred.view(batch_size, -1, self.height, self.max_size).to(feature.device)
        else:
            timestep_input = onehot_bos
            pred = torch.zeros(batch_size, self.max_size,
                               dtype=torch.int) + self.charset.blank
            for timestep in range(self.max_size):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    timestep_input, decoder_hidden, decoder_input, train=False)
                i = decoder_output.argmax(dim=1)
                timestep_input = i.detach()
                pred[:, timestep] = i
                if (i == self.charset.blank).all():
                    break
            return pred.to(feature.device)


class Attn(nn.Module):
    def __init__(self, method, hidden_dims, embed_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_dims = hidden_dims
        self.embed_size = embed_size
        self.attn = nn.Linear(2 * self.hidden_dims + embed_size, hidden_dims)
        # self.attn = nn.Linear(hidden_dims, hidden_dims)
        self.v = nn.Parameter(torch.rand(hidden_dims))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        H = hidden.repeat(max_len, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        # compute attention score
        attn_energies = self.score(H, encoder_outputs)
        # normalize with softmax
        return nn.functional.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # print('hidden: ', hidden.size())
        # print('encoder_outputs: ', encoder_outputs.size())
        # [B*T*2H]->[B*T*H]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        # energy = F.tanh(self.attn(hidden + encoder_outputs)) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2, 1)  # [B*H*T]
        v = self.v.repeat(
            encoder_outputs.data.shape[0], 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class AttentionRNNCell(nn.Module):  # Bahdanau attention based
    def __init__(self, hidden_dims, embedded_dims, nr_classes,
                 n_layers=1, dropout_p=0, bidirectional=False):
        super(AttentionRNNCell, self).__init__()

        self.hidden_dims = hidden_dims
        self.embedded_dims = embedded_dims
        self.nr_classes = nr_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(nr_classes, nr_classes)
        self.embedding.weight.data = torch.eye(nr_classes)
        self.dropout = nn.Dropout(dropout_p)
        self.word_linear = nn.Linear(nr_classes, hidden_dims)
        self.attn = Attn('concat', hidden_dims, embedded_dims)
        self.rnn = nn.GRUCell(2 * hidden_dims + embedded_dims, hidden_dims)
        self.out = nn.Linear(hidden_dims, nr_classes)

    def forward(self, word_input, last_hidden, encoder_outputs, train=True):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        '''

        # Get the embedding of the current input word (last output word)
        batch_size = word_input.size(0)
        word_embedded_onehot = self.embedding(word_input.to(last_hidden.device).type(
            torch.long)).view(1, batch_size, -1).to(last_hidden.device)  # (1,N,V)
        word_embedded = self.word_linear(word_embedded_onehot)  # (1, N, H)

        # Calculate attention weights and apply to encoder outputs
        # print('pre encoder_outputs: ', encoder_outputs.size())
        attn_weights = self.attn(last_hidden, encoder_outputs)
        context = attn_weights.bmm(
            encoder_outputs.transpose(0, 1))  # (N, 1, V)
        context = context.transpose(0, 1)  # (1, N, V)

        rnn_input = torch.cat([word_embedded, context], 2)
        last_hidden = last_hidden.view(batch_size, -1)
        rnn_input = rnn_input.view(batch_size, -1)
        hidden = self.rnn(rnn_input, last_hidden)

        if train:
            output = nn.functional.log_softmax(self.out(hidden), dim=1)
        else:
            output = nn.functional.softmax(self.out(hidden), dim=1)
        return output, hidden, attn_weights
