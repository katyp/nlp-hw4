#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from collections import namedtuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pdb

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):
    """ Simple Neural Machine Translation Model:
        - Bidrectional LSTM Encoder
        - Unidirection LSTM Decoder
        - Global Attention Model (Luong, et al. 2015)
    """
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2):
        """ Init NMT Model.

        ********** IMPORTANT ***********
        If you add parameters here for problem 2, be sure to read and understand the `save` and `load`
        methods, and the modify them as needed to make sure your model is saved correctly.

        @param embed_size (int): Embedding size (dimensionality)
        @param hidden_size (int): Hidden Size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        @param dropout_rate (float): Dropout probability, for attention

        """
        super(NMT, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        # default values
        self.source_embeddings = None
        self.target_embeddings = None
        self.encoder = None
        self.decoder = None
        self.h_projection = None
        self.c_projection = None
        self.att_projection = None
        self.combined_output_projection = None
        self.target_vocab_projection = None
        self.dropout = None

        src_pad_token_idx = vocab.src['<pad>']
        self.source_embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=src_pad_token_idx)

        tgt_pad_token_idx = vocab.tgt['<pad>']
        self.target_embeddings = nn.Embedding(len(vocab.tgt), embed_size, padding_idx=tgt_pad_token_idx)

        # Bidirectional LSTM with bias
        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        # LSTM Cell with bia
        self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)

        # Linear Layer with no bias), called W_{h} in the PDF.
        self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Linear Layer with no bias), called W_{c} in the PDF.
        self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Linear Layer with no bias), called W_{attProj} in the PDF.
        self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # Linear Layer with no bias), called W_{u} in the PDF.
        self.combined_output_projection = nn.Linear(hidden_size * 2 + hidden_size, hidden_size, bias=False)
        # Linear Layer with no bias), called W_{vocab} in the PDF.
        self.target_vocab_projection = nn.Linear(hidden_size, len(vocab.tgt), bias=False)
        # Dropout Layer
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """ Take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences under the language models learned by the NMT system.

        @param source (List[List[str]]): list of source sentence tokens
        @param target (List[List[str]]): list of target sentence tokens, wrapped by `<s>` and `</s>`

        @returns scores (Tensor): a variable/tensor of shape (b, ) representing the
                                    log-likelihood of generating the gold-standard target sentence for
                                    each example in the input batch. Here b = batch size.
        """
        # Compute sentence lengths
        source_lengths = [len(s) for s in source]

        # Convert list of lists into tensors
        source_padded = self.vocab.src.to_input_tensor(source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(target, device=self.device)   # Tensor: (tgt_len, b)

        #     Run the network forward:
        #     1. Apply the encoder to `source_padded` by calling `self.encode()`
        #     2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        #     3. Apply the decoder to compute combined-output by calling `self.decode()`
        #     4. Compute log probability distribution over the target vocabulary using the
        #        combined_outputs returned by the `self.decode()` function.

        enc_hiddens, dec_init_state = self.encode(source_padded, source_lengths)
        enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        combined_outputs = self.decode(enc_hiddens, enc_masks, dec_init_state, target_padded)
        P = F.log_softmax(self.target_vocab_projection(combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = \
            torch.gather(P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)
        return scores

    def _initialize_hiddens():
        return (torch.zeroes(self.hidden_size), torch.zeroes(self.hidden_size))
    def _h_i_enc(i, hidden_states):
        # forward is i*2, bwd is i*2 + 1
        i_fwd = 2*i
        i_bwd = 2*i + 1
        # [batch x hsize]
        fwd = hidden_states[i_fwd]
        bwd = hidden_states[i_fwd]

        h_i_enc = nn.cat((fwd, bwd), 1)
        return h_i_enc

    def encode(self, source_padded: torch.Tensor,
               source_lengths: List[int]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """ Apply the encoder to source sentences to obtain encoder hidden states.
            Additionally, take the final states of the encoder and project them to obtain initial states for decoder.

        @param source_padded (Tensor): Tensor of padded source sentences with shape (src_len, b), where
                                        b = batch_size, src_len = maximum source sentence length. Note that
                                       these have already been sorted in order of longest to shortest sentence.
        @param source_lengths (List[int]): List of actual lengths for each of the source sentences in the batch
        @returns enc_hiddens (Tensor): Tensor of hidden units with shape (b, src_len, h*2), where
                                        b = batch size, src_len = maximum source sentence length, h = hidden size.
        @returns dec_init_state (tuple(Tensor, Tensor)): Tuple of tensors representing the decoder's initial
                                                hidden state and cell.
        """
        enc_hiddens, dec_init_state = None, None

        # YOUR CODE HERE (~ 8 Lines)
        # TODO:
        #     1. Construct Tensor `X` of source sentences with shape (src_len, b, e) using the source model embeddings,
        #         here src_len = maximum source sentence length, b = batch size, e = embedding size. Note
        #         that there is no initial hidden state or cell for the decoder.
        #     2. Compute `enc_hiddens`, `last_hidden`, `last_cell` by applying the `self.encoder` to `X`.
        #         - This will require packing X and then unpacking the output, see the documentation nn.LSTM
        #     3. Compute `dec_init_state` = (init_decoder_hidden, init_decoder_cell):
        #         - `last_hidden` and `last_cell` will need to be modified so that they have shape
        #            (b, 2*h), and then h_projection and c_projection should be applied
        #
        # See the following docs, as you may need to use some of the following functions in your implementation:
        #     Pack the padded sequence X before passing to the encoder:
        #         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pack_padded_sequence
        #     Pad the packed sequence, enc_hiddens, returned by the encoder:
        #         https://pytorch.org/docs/stable/nn.html#torch.nn.utils.rnn.pad_packed_sequence
        #     Tensor Concatenation:
        #         https://pytorch.org/docs/stable/torch.html#torch.cat
        #     Tensor Permute:
        #         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.permute

        pdb.set_trace()
        src_len, b = source_padded.size()
        X = self.source_embeddings(source_padded)

        h0, c0 = _initialize_hiddens()
        # [src_len x b x e] --> packed_sequence
        # input should be src_len x b x dimension (*=dim=self.embedding_size)
        # lengths: source_lengths
        packed_input = torch.nn.pack_padded_sequence(X, source_lengths)
        # do some unpacking I guess. encoder is an LSTM, so:
        # seq_len, batch, input_size
        output, (hn, cn) = self.encoder(packed_input, (h0, c0))
        padded_output = torch.nn.pad_packed_sequence(output, source_lengths)

        # self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=True)

        # hn: (num_layers * 2, batch, hidden_size)
        # cn: (num_layers * 2, batch, hidden_size)
        #   -- hn, cn: for every layer (first to last): 5 batches, hsize say 20
        # output: seq_len, batch, hidden_size
        #   -- output: sentence length, 5 batches, hsize say 20

        num_sentences = len(source_lengths)

        enc_hiddens = hn            # [per-layer x b x hsize]
        init_decoder_hidden = _h_i_enc(hn, num_sentences - 2)    # [b x hsize] make last_hidden for *each* sentence in *each* batch
        init_decoder_cell = _h_i_enc(cn, num_sentences - 2)

        dec_init_state = (init_decoder_hidden, init_decoder_cell)

        # END YOUR CODE

        return enc_hiddens, dec_init_state

    def decode(self, enc_hiddens: torch.Tensor, enc_masks: torch.Tensor,
               dec_init_state: Tuple[torch.Tensor, torch.Tensor],
               target_padded: torch.Tensor) -> torch.Tensor:
        """Compute combined output vectors for a batch.

        @param enc_hiddens (Tensor): Hidden states (b, src_len, h*2), where
                                     b = batch size, src_len = maximum source sentence length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks (b, src_len), where
                                     b = batch size, src_len = maximum source sentence length.
        @param dec_init_state (tuple(Tensor, Tensor)): Initial state and cell for decoder
        @param target_padded (Tensor): Gold-standard padded target sentences (tgt_len, b), where
                                       tgt_len = maximum target sentence length, b = batch size.

        @returns combined_outputs (Tensor): combined output tensor  (tgt_len, b,  h), where
                                        tgt_len = maximum target sentence length, b = batch_size,  h = hidden size
        """
        # Chop of the <END> token for max length sentences.
        target_padded = target_padded[:-1]

        # Initialize the decoder state (hidden and cell)
        dec_state = dec_init_state

        # Initialize previous combined output vector o_{t-1} as zero
        batch_size = enc_hiddens.size(0)
        o_prev = torch.zeros(batch_size, self.hidden_size, device=self.device)

        # Initialize a list we will use to collect the combined output o_t on each step
        combined_outputs = []

        # YOUR CODE HERE (~9 Lines)
        # TODO:
        #     1. Apply the attention projection layer to `enc_hiddens` to obtain `enc_hiddens_proj`,
        #         which should be shape (b, src_len, h),
        #         where b = batch size, src_len = maximum source length, h = hidden size.
        #         This is applying W_{attProj} to h^enc, as described in the PDF.
        #     2. Construct tensor `Y` of target sentences with shape (tgt_len, b, e) using the target model embeddings.
        #         where tgt_len = maximum target sentence length, b = batch size, e = embedding size.
        #     3. Use the torch.split function to iterate over the time dimension of Y.
        #         Within the loop, this will give you Y_t of shape (1, b, e) where b = batch size, e = embedding size.
        #             - Construct Ybar_t by concatenating Y_t with o_prev.
        #             - Use the step function to compute the the Decoder's next (cell, state) values
        #               as well as the new combined output o_t.
        #             - Append o_t to combined_outputs
        #             - Update o_prev to the new o_t.
        #     4. Use torch.stack to convert combined_outputs from a list length tgt_len of
        #         tensors shape (b, h), to a single tensor shape (tgt_len, b, h)
        #         where tgt_len = maximum target sentence length, b = batch size, h = hidden size.
        #
        # Use the following docs to implement this functionality:
        #     Tensor Splitting (iteration):
        #         https://pytorch.org/docs/stable/torch.html#torch.split
        #     Tensor Dimension Squeezing:
        #         https://pytorch.org/docs/stable/torch.html#torch.squeeze
        #     Tensor Concatenation:
        #         https://pytorch.org/docs/stable/torch.html#torch.cat
        #     Tensor Stacking:
        #         https://pytorch.org/docs/stable/torch.html#torch.stack

        # END YOUR CODE
        return combined_outputs

    def step(self, Ybar_t: torch.Tensor,
             dec_state: Tuple[torch.Tensor, torch.Tensor],
             enc_hiddens: torch.Tensor,
             enc_hiddens_proj: torch.Tensor,
             enc_masks: torch.Tensor) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """ Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h),
                where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h.
                Tensor is with shape (b, src_len, h),
                where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length.

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h),
                where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h),
                where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """

        combined_output = None

        e_t = None
        # YOUR CODE HERE (~3 Lines)
        # TODO:
        #     1. Apply the decoder to `Ybar_t` and `dec_state`to obtain the new dec_state.
        #     2. Split dec_state into its two parts (dec_hidden, dec_cell)
        #     3. Compute the attention scores e_t, a Tensor shape (b, src_len).
        #        Note: b = batch_size, src_len = maximum source length, h = hidden size.
        #
        #       Hints:
        #         - dec_hidden is shape (b, h) and corresponds to h^dec_t in the PDF (batched)
        #         - enc_hiddens_proj is shape (b, src_len, h) and corresponds to W_{attProj} h^enc (batched).
        #         - Use batched matrix multiplication (torch.bmm) to compute e_t.
        #         - To get the tensors into the right shapes for bmm, you'll need to do some squeezing and unsqueezing.
        #         - When using the squeeze() function make sure to specify the dimension you want to squeeze
        #             over. Otherwise, you will remove the batch dimension accidentally, if batch_size = 1.
        #
        # Use the following docs to implement this functionality:
        #     Batch Multiplication:
        #        https://pytorch.org/docs/stable/torch.html#torch.bmm
        #     Tensor Unsqueeze:
        #         https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
        #     Tensor Squeeze:
        #         https://pytorch.org/docs/stable/torch.html#torch.squeeze

        # END YOUR CODE

        # Set e_t to -inf where enc_masks has 1
        if enc_masks is not None:
            e_t.data.masked_fill_(enc_masks.bool(), -float('inf'))

        # YOUR CODE HERE (~6 Lines)
        # TODO:
        #     1. Apply softmax to e_t to yield alpha_t
        #     2. Use batched matrix multiplication between alpha_t and enc_hiddens to obtain the
        #         attention output vector, a_t.
        #     Hints:
        #           - alpha_t is shape (b, src_len)
        #           - enc_hiddens is shape (b, src_len, 2h)
        #           - a_t should be shape (b, 2h)
        #           - You will need to do some squeezing and unsqueezing.
        #     Note: b = batch size, src_len = maximum source length, h = hidden size.
        #     3. Concatenate dec_hidden with a_t to compute tensor U_t
        #     4. Use the output projection layer to compute tensor V_t
        #     5. Compute tensor O_t using the Tanh function and the dropout layer.
        #
        # Use the following docs to implement this functionality:
        #     Softmax:
        #         https://pytorch.org/docs/stable/nn.html#torch.nn.functional.softmax
        #     Batch Multiplication:
        #        https://pytorch.org/docs/stable/torch.html#torch.bmm
        #     Tensor View:
        #         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view
        #     Tensor Concatenation:
        #         https://pytorch.org/docs/stable/torch.html#torch.cat
        #     Tanh:
        #         https://pytorch.org/docs/stable/torch.html#torch.tanh

        # END YOUR CODE

        combined_output = O_t
        return dec_state, combined_output, e_t

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """ Generate sentence masks for encoder hidden states.

        @param enc_hiddens (Tensor): encodings of shape (b, src_len, 2*h), where b = batch size,
                                     src_len = max source length, h = hidden size.
        @param source_lengths (List[int]): List of actual lengths for each of the sentences in the batch.

        @returns enc_masks (Tensor): Tensor of sentence masks of shape (b, src_len),
                                    where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str],
                    beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(src_sents_var, [len(src_sent)])
        if self.att_projection is None:
            src_encodings_att_linear = None
        else:
            src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        hypotheses = [['<s>']]
        hyp_scores = torch.zeros(len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            # Build a copy of `src_encodings` and `src_encodings_att_linear` for each
            # hypothesis in our beam
            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            if src_encodings_att_linear is None:
                exp_src_encodings_att_linear = None
            else:
                exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                               src_encodings_att_linear.size(1),
                                                                               src_encodings_att_linear.size(2))

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]] for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.target_embeddings(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1, exp_src_encodings,
                                                exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            hyp_scores = torch.tensor(new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        """ Determine which device to place the Tensors upon, CPU or GPU.
        """
        return self.source_embeddings.weight.device

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        # This will re-build the NMT model with
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """

        params = {
            'args': dict(embed_size=self.embed_size,
                         hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
