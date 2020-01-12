# coding=utf-8
import pickle
import sys
from components.dataset import Dataset, Example
import subprocess

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

if __name__ == '__main__':
    print(sys.argv)
    eval()