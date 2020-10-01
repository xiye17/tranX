# coding=utf-8
from __future__ import print_function

import argparse
from itertools import chain

import six.moves.cPickle as pickle
from six.moves import xrange as range
from six.moves import input
import traceback

import numpy as np
import time
import os
import sys

import torch
from torch.autograd import Variable

import evaluation
from asdl import *
from asdl.asdl import ASDLGrammar
from common.registerable import Registrable
from components.dataset import Dataset, Example
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils, utils

from model.parser import Parser
from model.utils import GloveHelper, get_parser_class
from eval import eval_streg_predictions
from asdl.lang.streg.streg_transition_system import partial_asdl_ast_to_streg_ast

if six.PY3:
    # import additional packages for wikisql dataset (works only under Python 3)
    from model.wikisql.dataset import WikiSqlExample, WikiSqlTable, TableColumn
    from model.wikisql.parser import WikiSqlParser
    from datasets.wikisql.dataset import Query, DBEngine


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def train(args):
    """Maximum Likelihood Estimation"""

    # load in train/dev set
    train_set = Dataset.from_bin_file(args.train_file)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else: dev_set = Dataset(examples=[])
    train_set.examples = train_set.examples[:200]
    dev_set.examples = dev_set.examples[:50]
    vocab = pickle.load(open(args.vocab, 'rb'))

    grammar = ASDLGrammar.from_text(open(args.asdl_file).read())
    transition_system = Registrable.by_name(args.transition_system)(grammar)

    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    model = parser_cls(args, vocab, transition_system)
    model.train()

    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    if args.cuda: model.cuda()

    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        nn_utils.uniform_init(-args.uniform_init, args.uniform_init, model.parameters())
    elif args.glorot_init:
        print('use glorot initialization', file=sys.stderr)
        nn_utils.glorot_init(model.parameters())

    # load pre-trained word embedding (optional)
    if args.glove_embed_path:
        print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
        glove_embedding = GloveHelper(args.glove_embed_path)
        glove_embedding.load_to(model.src_embed, vocab.source)

    print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    while True:
        epoch += 1
        epoch_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
            batch_examples = [e for e in batch_examples if len(e.tgt_actions) <= args.decode_max_time_step]
            train_iter += 1
            optimizer.zero_grad()

            ret_val = model.score(batch_examples)
            loss = -ret_val[0]

            # print(loss.data)
            loss_val = torch.sum(loss).data.item()
            report_loss += loss_val
            report_examples += len(batch_examples)
            loss = torch.mean(loss)

            if args.sup_attention:
                att_probs = ret_val[1]
                if att_probs:
                    sup_att_loss = -torch.log(torch.cat(att_probs)).mean()
                    sup_att_loss_val = sup_att_loss.data[0]
                    report_sup_att_loss += sup_att_loss_val

                    loss += sup_att_loss

            loss.backward()

            # clip gradient
            if args.clip_grad > 0.:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()

            if train_iter % args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if args.sup_attention:
                    log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                    report_sup_att_loss = 0.

                print(log_str, file=sys.stderr)
                report_loss = report_examples = 0.

        print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            model_file = args.save_to + '.iter%d.bin' % train_iter
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                                    epoch, eval_results,
                                    evaluator.default_metric,
                                    dev_score,
                                    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            patience = 0
            model_file = args.save_to + '.bin'
            print('save the current model ..', file=sys.stderr)
            print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')
        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            print('reached max epoch, stop!', file=sys.stderr)
            exit(0)

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                print('early stop!', file=sys.stderr)
                exit(0)

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                print('reset optimizer', file=sys.stderr)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    assert args.load_model

    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))



def pl_test(args):
    test_set = Dataset.from_bin_file(args.test_file)
    print(max([len(x.tgt_actions) for x in test_set]))
    # exit()
    assert args.load_model
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    eval_results, decode_results = evaluation.pl_evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    print(eval_results, file=sys.stderr)
    if args.save_decode_to:
        pickle.dump(decode_results, open(args.save_decode_to, 'wb'))

# non consistent
# debug_idx = [2, 4, 14, 16, 20, 28, 32, 37, 39, 43, 47, 56, 57, 58, 59, 65, 66, 67, 68, 73, 75, 80, 85, 86, 87, 91, 98, 100, 101, 104, 109, 123, 126, 130, 131, 141, 146, 150, 156, 166, 172, 173, 181, 184, 191, 193, 198, 203, 209, 211, 212, 216, 224, 226, 227, 228, 234, 236, 237, 239, 247, 249, 256, 258, 260, 265, 266, 267, 268, 273, 274, 286, 289, 295, 296, 298, 299, 307, 309, 319, 346, 348, 353, 359, 371, 385, 386, 389, 398, 401, 414, 415, 416, 417, 418, 421, 424, 425, 426, 429, 443, 444, 449, 452, 454, 459, 468, 469, 470, 471, 479, 481, 488, 491, 492, 493, 494, 498, 501, 503, 509, 510, 515, 517, 519, 520, 525, 529, 537, 543, 544, 548, 553, 559, 574, 598, 606, 609, 612, 613, 621, 627]

# spurious
# debug_idx = [5, 9, 11, 13, 19, 27, 36, 38, 41, 54, 62, 64, 69, 70, 79, 81, 84, 89, 90, 94, 97, 99, 107, 108, 115, 118, 120, 125, 127, 128, 132, 133, 137, 140, 143, 145, 147, 148, 154, 155, 157, 158, 159, 160, 161, 164, 165, 170, 171, 179, 182, 195, 196, 204, 217, 218, 220, 223, 231, 233, 235, 254, 255, 269, 271, 272, 275, 277, 288, 294, 297, 306, 308, 311, 312, 315, 318, 329, 330, 331, 332, 338, 340, 342, 349, 351, 352, 355, 356, 357, 358, 374, 375, 376, 392, 393, 396, 402, 403, 412, 413, 422, 423, 430, 441, 458, 460, 461, 462, 476, 478, 487, 489, 496, 497, 500, 502, 508, 513, 514, 516, 522, 523, 526, 527, 528, 532, 533, 538, 539, 540, 542, 546, 547, 551, 552, 554, 555, 556, 558, 562, 568, 571, 572, 575, 576, 577, 578, 579, 583, 594, 599, 607, 608, 611, 614, 617, 618, 623]

debug_idx = [2, 5]
def pl_debug(args):
    test_set = Dataset.from_bin_file(args.test_file)
    test_set.examples = [x for i,x in enumerate(test_set.examples) if i in debug_idx]
    assert args.load_model
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()
    evaluator = Registrable.by_name(args.evaluator)(transition_system, args=args)
    
    # decode_results, turning_point = [before_decodes, after_decodes], 
    eval_results, decode_results, debug_info = evaluation.pl_evaluate(test_set.examples, parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True, debug=True)
    print(eval_results, file=sys.stderr)
    # if args.save_decode_to:
    #     pickle.dump(decode_results, open(args.save_decode_to, 'wb'))

    # dump_debug_info

    with open("debug_info.txt", "w") as f:
        for idx, ex, pred_hyps, info in zip(debug_idx, test_set.examples, decode_results, debug_info):
            predictions = [x.code.replace(" ", "") for x in pred_hyps]
            f.write("----------------{}------------------\n".format(idx))
            f.write("Src: {}\n".format(" ".join(ex.src_sent)))
            f.write("Tgt: {}\n".format(ex.tgt_code.replace(" ", "")))
            f.write("Predictions:\n")
            pred_results = eval_streg_predictions(predictions, ex)
            for p, r in zip(predictions, pred_results):
                f.write("\t{} {}\n".format(r, p.replace(" ", "")))

            if info is None:
                f.write("\n")
                continue
            prev_beam, latter_beam = info
            prev_beam.sort(key=lambda hyp: -hyp.score)
            latter_beam.sort(key=lambda hyp: -hyp.score)
            f.write("Beam {}:\n".format(prev_beam[0].t))
            for p_hyp in prev_beam:
                _, partial_ast = partial_asdl_ast_to_streg_ast(p_hyp.tree)
                f.write("\t{:.2f} {}\n".format(p_hyp.score, partial_ast.debug_form()))
            
            f.write("Beam {}:\n".format(latter_beam[0].t))
            for p_hyp in latter_beam:
                _, partial_ast = partial_asdl_ast_to_streg_ast(p_hyp.tree)
                f.write("\t{:.2f} {}\n".format(p_hyp.score, partial_ast.debug_form()))
            f.write("\n")


def easy_pickle_read(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def easy_pickle_dump(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

from components.cache import *
from components.result import *
from synthesizer.synthesizer import NoPruneSynthesizer
from eval import batch_filtering_test
from tqdm import tqdm


def synthesize(args):
    test_set = Dataset.from_bin_file(args.test_file)
    test_set.examples = test_set.examples
    print(max([len(x.tgt_actions) for x in test_set]))
    # exit()
    assert args.load_model
    print('load model from [%s]' % args.load_model, file=sys.stderr)
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = args.cuda
    # set the correct domain from saved arg
    args.lang = saved_args.lang

    parser_cls = Registrable.by_name(args.parser)
    parser = parser_cls.load(model_path=args.load_model, cuda=args.cuda)
    parser.eval()

    # test_set.examples = [test_set.examples[i] for i in poi]
    parser.eval()
    cache = SynthCache.from_file("misc/cache.pkl")

    synthesizer = NoPruneSynthesizer(args, parser, score_func='prob')
    # with torch.no_grad():
    synth_results = []
    budgets_used = []
    for ex in tqdm(test_set, desc='Synthesize', file=sys.stdout, total=len(test_set)):
        result, num_exec = synthesizer.solve(ex, cache=cache)
        synth_results.append(result)
        budgets_used.append(num_exec)
    act_tree_to_ast = lambda x: parser.transition_system.build_ast_from_actions(x)
    pred_codes = [[parser.transition_system.ast_to_surface_code(x.tree) for x in preds] for preds in synth_results]
    top_codes = [x[0] if x else "" for x in pred_codes]
    match_results = [ e.tgt_code == r for e, r in zip(test_set, top_codes)]
    match_acc = sum(match_results) * 1. / len(match_results)

    results = []
    acc = 0
    for pred_hyps, gt_exs in zip(pred_codes, test_set):
        # top_pred = pred_hyps[0]
        codes = [x.replace(" ", "") for x in pred_hyps]
        gt_code = gt_exs.tgt_code.replace(" ", "")
        # print(codes)
        # print(gt_code)
        # exit()

        match_result = batch_filtering_test(gt_code, codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1
    cache.dump()
    print("Eval Acc", match_acc)
    print("Oracle Acc", acc * 1.0/len(test_set) )

    eval_results = [SynthResult(progs, budget, result) for (progs, budget, result) in zip(synth_results, budgets_used, results)]
    easy_pickle_dump(eval_results, args.save_decode_to)


if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        pl_test(args)
        # pl_debug(args)
    elif args.mode == 'synth':
        synthesize(args)
    else:
        raise RuntimeError('unknown mode')
