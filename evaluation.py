# coding=utf-8
from __future__ import print_function

import sys
import traceback
from tqdm import tqdm


def decode(examples, model, args, verbose=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()

    is_wikisql = args.parser == 'wikisql_parser'

    decode_results = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if is_wikisql:
            hyps = model.parse(example.src_sent, context=example.table, beam_size=args.beam_size)
        else:
            hyps = model.parse(example.src_sent, context=None, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             example.tgt_code,
                                                                                             hyp_id,
                                                                                             hyp.tree.to_string()), file=sys.stdout)
                    if got_code:
                        print()
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)

    if was_training: model.train()

    return decode_results


def evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False):
    decode_results = decode(examples, parser, args, verbose=verbose)

    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)

    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result

def pl_decode(examples, model, args, verbose=False, debug=False, **kwargs):
    ## TODO: create decoder for each dataset

    if verbose:
        print('evaluating %d examples' % len(examples))

    was_training = model.training
    model.eval()


    decode_results = []
    debug_info = []
    count = 0
    for example in tqdm(examples, desc='Decoding', file=sys.stdout, total=len(examples)):
        if debug:
            hyps, debug_output = model.pl_parse(example.src_sent, context={'str_exs': example.meta['str_exs'], 'const_map': example.meta['const_map'], 'tgt_ast': example.tgt_ast}, beam_size=args.beam_size, pl_debug=debug)
        else:
            hyps = model.pl_parse(example.src_sent, context={'str_exs': example.meta['str_exs'], 'const_map': example.meta['const_map']}, beam_size=args.beam_size)
        decoded_hyps = []
        for hyp_id, hyp in enumerate(hyps):
            got_code = False
            try:
                hyp.code = model.transition_system.ast_to_surface_code(hyp.tree)
                got_code = True
                decoded_hyps.append(hyp)
            except:
                if verbose:
                    print("Exception in converting tree to code:", file=sys.stdout)
                    print('-' * 60, file=sys.stdout)
                    print('Example: %s\nIntent: %s\nTarget Code:\n%s\nHypothesis[%d]:\n%s' % (example.idx,
                                                                                             ' '.join(example.src_sent),
                                                                                             example.tgt_code,
                                                                                             hyp_id,
                                                                                             hyp.tree.to_string()), file=sys.stdout)
                    if got_code:
                        print()
                        print(hyp.code)
                    traceback.print_exc(file=sys.stdout)
                    print('-' * 60, file=sys.stdout)

        count += 1

        decode_results.append(decoded_hyps)
        if debug:
            debug_info.append(debug_output)

    if was_training: model.train()
    if debug:
        return decode_results, debug_info        
    return decode_results

def pl_evaluate(examples, parser, evaluator, args, verbose=False, return_decode_result=False, eval_top_pred_only=False, debug=False):
    if debug:
        decode_results, turning_point = pl_decode(examples, parser, args, verbose=verbose, debug=debug)
    else:
        decode_results = pl_decode(examples, parser, args, verbose=verbose, debug=debug)
    eval_result = evaluator.evaluate_dataset(examples, decode_results, fast_mode=eval_top_pred_only)
    
    if debug:
        return eval_result, decode_results, turning_point
    if return_decode_result:
        return eval_result, decode_results
    else:
        return eval_result
