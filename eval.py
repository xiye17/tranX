# coding=utf-8
import pickle
import sys
from components.dataset import Dataset, Example
import subprocess
from os.path import join
import os
import random

def check_equiv(spec0, spec1):
    if spec0 == spec1:
        # print("exact", spec0, spec1)
        return True
    # try:
    out = subprocess.check_output(
        ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'equiv',
            spec0, spec1], stderr=subprocess.DEVNULL)
    out = out.decode("utf-8")
    out = out.rstrip()
    # if out == "true":
    #     print("true", spec0, spec1)

    return out == "true"

def post_process(x):
    x = x.replace("<m0>", "<!>")
    x = x.replace("<m1>", "<@>")
    x = x.replace("<m2>", "<#>")
    x = x.replace("<m3>", "<$>")
    x = x.replace(" ", "")
    return x

def eval():
    test_set = Dataset.from_bin_file(sys.argv[1])
    # test_set.examples = test_set.examples[:200]
    decodes = pickle.load(open(sys.argv[2], "rb"))
    results = []
    acc = 0
    for i, (pred_hyps, gt_exs) in enumerate(zip(decodes, test_set)):
        top_pred = pred_hyps[0]
        gt_code = post_process(gt_exs.tgt_code)
        pred_code = post_process(top_pred.code)
    
        eq_res = check_equiv(pred_code, gt_code)
        results.append(eq_res)
        acc += eq_res
        print(acc, i)
    print(sum(results))


def external_evalute_single(gt_spec, preds, exs, flag_force=False):
    pred_line = " ".join(preds)
    exs_line = " ".join(["{},{}".format(x[0], x[1]) for x in exs])
    flag_str = "true" if flag_force else "false"

    flag_use_file = len(preds) > 100
    if flag_use_file:
        filename = join("./external/", "eval_single_{}.in".format(random.random()))
        with open(filename, "w") as f:
            f.write(pred_line + "\n")
            f.write(exs_line + "\n")
            f.write(gt_spec)
        out = subprocess.check_output(
            ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single_file',
                filename, flag_str])
        os.remove(filename)
    else:
        try:
            out = subprocess.check_output(
                ['java', '-cp', './external/datagen.jar:./external/lib/*', '-ea', 'datagen.Main', 'evaluate_single',
                    pred_line, exs_line, gt_spec, flag_str],timeout=20)
        except subprocess.TimeoutExpired as e:
            print(e)
            return 'invalid', ['invalid']
            

    
    out = out.decode("utf-8")
    out = out.rstrip()
    vals = out.split(" ")
    return vals[0], vals[1:]

def inverse_regex_with_map(r, maps):
    for m in maps:
        src = m[0]
        if len(m[1]) == 1:
            dst = "<{}>".format(m[1])
        else:
            dst = "const(<{}>)".format(m[1])
        r = r.replace(src, dst)
    return r


def batch_filtering_test(gt, preds, meta, flag_force=False):
    gt = inverse_regex_with_map(gt, meta["const_map"])
    preds = [inverse_regex_with_map(x, meta["const_map"]) for x in preds]

    global_res, pred_res = external_evalute_single(gt, preds, meta["str_exs"], flag_force)
    if global_res in ["exact", "equiv"]:
        return True, global_res, pred_res
    else:
        return False, global_res, pred_res


def eval_streg():
    test_set = Dataset.from_bin_file(sys.argv[1])
    # test_set.examples = test_set.examples[:200]
    decodes = pickle.load(open(sys.argv[2], "rb"))
    decodes = [x.decodes for x in decodes]
    results = []
    acc = 0
    for idx, (pred_hyps, gt_exs) in enumerate(zip(decodes, test_set)):
        # top_pred = pred_hyps[0]
        pred_codes = [x.code.replace(" ", "") for x in pred_hyps]
        gt_code = gt_exs.tgt_code.replace(" ", "")

        match_result = batch_filtering_test(gt_code, pred_codes, gt_exs.meta, flag_force=True)
        results.append(match_result)
        if match_result[0]:
            acc += 1
        print("{}----{}/{}".format(acc, idx, len(decodes)))

    with open("testi-first50-report.txt", "w") as f:
        for i, res in enumerate(results):
            line_fields = [str(i), str(res[0]), str(res[1])]
            line_fields.extend(["{},{}".format(x[0], x[1])
                                for x in enumerate(res[2])])
            f.write(" ".join(line_fields) + "\n")

def eval_streg_predictions(predictions, gt_exs):
    gt_code = gt_exs.tgt_code.replace(" ", "")
    match_result = batch_filtering_test(gt_code, predictions, gt_exs.meta, flag_force=True)
    return match_result[2]

if __name__ == '__main__':
    print(sys.argv)
    # eval()
    eval_streg()