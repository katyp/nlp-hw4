#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    python main.py --mode train --model_path <model_path>
    python main.py --mode eval_dev --model_path <model_path> --output_file <output_file>
    python main.py --mode eval_test --model_path <model_path> --output_file <output_file>



Options:
   See python main.py --help
"""

import math
import sys
import time
import argparse
from os import makedirs
from os.path import dirname, exists

from nmt_model import Hypothesis, NMT
from nltk.translate.bleu_score import corpus_bleu
import numpy as np
from typing import List
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab

import torch
import torch.nn.utils


def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    @param references (List[List[str]]): a list of gold-standard reference target sentences
    @param hypotheses (List[Hypothesis]): a list of hypotheses, one for each reference
    @returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args):
    """ Train the NMT Model.
    """
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    vocab = Vocab.load(args.vocab_file)
    model = NMT(embed_size=args.embed_size,
                hidden_size_enc=args.hidden_size_enc,
                hidden_size_dec=args.hidden_size_dec,
                vocab=vocab,
                attention_function_name=args.attention,
                dropout_rate=args.dropout
                )
    model.train()

    if np.abs(args.uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init))
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")
    print('use device: %s' % device)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        batch_num = math.ceil(len(train_data) / args.batch_size)
        current_iter = 0
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=args.batch_size, shuffle=True):
            current_iter += 1
            train_iter += 1

            optimizer.zero_grad()
            batch_size = len(src_sents)
            example_losses = -model(src_sents, tgt_sents) # value, log likelihood of target
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            loss.backward()

            # clip gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            # omitting leading `<s>`
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d (%d / %d), iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' %
                      (epoch, current_iter, batch_num, train_iter,
                       report_loss / report_examples,
                       math.exp(report_loss / report_tgt_words),
                       cum_examples,
                       report_tgt_words / (time.time() - train_time),
                       time.time() - begin_time))

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                      cum_loss / cum_examples,
                      np.exp(cum_loss / cum_tgt_words),
                      cum_examples))

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl))

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('epoch %d, iter %d: save currently the best model to [%s]' %
                          (epoch, train_iter, args.model_path))
                    model.save(args.model_path)
                    torch.save(optimizer.state_dict(), args.model_path + '.optim')
                elif patience < args.patience:
                    patience += 1
                    print('hit patience %d' % patience)

                    if patience == args.patience:
                        num_trial += 1
                        print('hit #%d trial' % num_trial)
                        if num_trial == args.max_num_trial:
                            print('early stop!')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                        print('load previously best model and decay learning rate to %f' % lr)

                        # load model
                        params = torch.load(args.model_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers')
                        optimizer.load_state_dict(torch.load(args.model_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == args.max_epoch:
                print('reached maximum number of epochs!')
                return


def decode(args, test_data_src, test_data_tgt=None):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    print("load test source sentences from [{}]".format(test_data_src))
    test_data_src = read_corpus(test_data_src, source='src')
    if test_data_tgt:
        print("load test target sentences from [{}]".format(test_data_tgt))
        test_data_tgt = read_corpus(test_data_tgt, source='tgt')

    print("load model from {}".format(args.model_path))
    model = NMT.load(args.model_path)

    if args.cuda:
        model = model.to(torch.device("cuda:0"))

    hypotheses = beam_search(model, test_data_src,
                             beam_size=args.beam_size,
                             max_decoding_time_step=args.max_decoding_time_step)

    if args.test_tgt:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100))

    if args.output_file:
        print("Saving predictions to " + args.output_file)
        with open(args.output_file, 'w') as f:
            for src_sent, hyps in zip(test_data_src, hypotheses):
                top_hyp = hyps[0]
                hyp_sent = ' '.join(top_hyp.value)
                f.write(hyp_sent + '\n')
    else:
        print("No output_file given, not saving predictions")


def beam_search(model: NMT, test_data_src: List[List[str]],
                beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    """ Run beam search to construct hypotheses for a list of src-language sentences.
    @param model (NMT): NMT Model
    @param test_data_src (List[List[str]]): List of sentences (words) in source language, from test set.
    @param beam_size (int): beam_size (# of hypotheses to hold for a translation at every step)
    @param max_decoding_time_step (int): maximum sentence length that Beam search can produce
    @returns hypotheses (List[List[Hypothesis]]): List of Hypothesis translations for every source sentence.
    """
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding'):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size,
                                             max_decoding_time_step=max_decoding_time_step)
            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def main():
    # Check pytorch version
    assert(torch.__version__ in {"1.3.0", "1.3.1"}), \
        "Please update your installation of PyTorch. " \
        "You have {} and you should have version 1.3.0 or 1.3.1".format(torch.__version__)
    assert(sys.version_info >= (3, 5)), \
        "Please update your installation of Python to version >= 3.5"

    parser = argparse.ArgumentParser()
    # Parameters that need to be set
    parser.add_argument("--mode",
                        choices=["train", "eval_dev", "eval_test", "eval_train"], required=True)
    parser.add_argument("--model_path", required=True)

    # Data locations, these shouldn't need to be changed from the defaults
    parser.add_argument("--train_src", default="data/train.fr", type=str,
                        help="train source file")
    parser.add_argument("--train_tgt", default="data/train.en", type=str,
                        help="train target file")
    parser.add_argument("--dev_src", default="data/dev.fr", type=str,
                        help="dev source file to use during training")
    parser.add_argument("--dev_tgt", default="data/dev.en", type=str,
                        help="dev target file to use during training")
    parser.add_argument("--test_src", default="data/test.fr", type=str,
                        help="test source file for using when decoding")
    parser.add_argument("--test_tgt", default="data/test.en", type=str,
                        help="test target file for using when decoding")
    parser.add_argument("--vocab_file", default="vocab.json", type=str)

    # Training hyperparameters
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--embed_size", default=64, type=int)
    parser.add_argument("--hidden_size_enc", default = 64, type=int)
    parser.add_argument("--hidden_size_dec", default = 64, type=int)
    parser.add_argument("--attention", default = "MULTIPLICATIVE", type=str)
    parser.add_argument("--clip_grad", default=5.0, type=float)
    parser.add_argument("--dropout", default=0.3, type=float)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--patience", default=5, type=int,
                        help="wait for how many iterations to decay learning rate")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--lr_decay", default=0.5, type=float, help="learning rate decay")
    parser.add_argument("--max_num_trial", default=5, type=int,
                        help="terminate training after how many trials")
    parser.add_argument("--uniform_init", default=0.1, type=float,
                        help="uniformly initialize all parameters")

    # Other training parameters
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--seed", default=123, type=int)
    parser.add_argument("--cuda", action='store_true', help="Use the GPU")
    parser.add_argument("--valid_niter", default=100, type=int,
                        help="perform validation after how many iterations")

    # Evaluation parameters
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--beam_size", default=5, type=int)
    parser.add_argument("--max_decoding_time_step", default=70, type=int,
                        help="maximum number of decoding time steps")
    args = parser.parse_args()

    if args.seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed * 13 // 7)

    if args.mode == "train":
        assert args.train_src
        assert args.train_tgt
        assert args.dev_src
        assert args.dev_tgt
        assert args.vocab_file
        if not exists(args.vocab_file):
            raise ValueError("A vocab file needs to be constructed before the model can be trained")

        model_dir = dirname(args.model_path)
        if model_dir and not exists(model_dir):
            makedirs(model_dir)
        train(args)
    elif args.mode == "eval_dev":
        assert args.dev_src
        decode(args, args.dev_src, args.dev_tgt)
    elif args.mode == "eval_train":
        assert args.train_src
        decode(args, args.train_src, args.train_tgt)
    elif args.mode == "eval_test":
        assert args.test_src
        decode(args, args.test_src, args.test_tgt)
    else:
        raise RuntimeError("invalid mode: {}".format(args.mode))


if __name__ == '__main__':
  main()
