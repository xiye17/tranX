
from os.path import join
import sys
import numpy as np
import pickle
from asdl.asdl import ASDLGrammar

from asdl.hypothesis import Hypothesis, ApplyRuleAction
from components.action_info import get_action_infos
from components.dataset import Example
from components.vocab import VocabEntry, Vocab

from asdl.lang.streg.streg_transition_system import *


def _read_lines(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [x.rstrip('\n') for x in lines]

    return lines

class StReg:
    @staticmethod
    def process_regex_dataset():
        grammar = ASDLGrammar.from_text(open('asdl/lang/streg/streg_asdl.txt').read())
        transition_system = StRegTransitionSystem(grammar)
        train_set = StReg.load_regex_dataset(transition_system, "train")
        val_set = StReg.load_regex_dataset(transition_system, "val")
        testi_set = StReg.load_regex_dataset(transition_system, "testi")
        teste_set = StReg.load_regex_dataset(transition_system, "teste")


        # generate vocabulary
        vocab_freq_cutoff = 2
        src_vocab = VocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)

        primitive_tokens = [map(lambda a: a.action.token,
                                filter(lambda a: isinstance(a.action, GenTokenAction), e.tgt_actions))
                            for e in train_set]
        primitive_vocab = VocabEntry.from_corpus(primitive_tokens, size=5000, freq_cutoff=0)

        # generate vocabulary for the code tokens!
        code_tokens = [transition_system.tokenize_code(e.tgt_code, mode='decoder') for e in train_set]
        code_vocab = VocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

        vocab = Vocab(source=src_vocab, primitive=primitive_vocab, code=code_vocab)
        print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

        action_len = [len(e.tgt_actions) for e in chain(train_set, testi_set, teste_set)]
        print('Max action len: %d' % max(action_len), file=sys.stderr)
        print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
        print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

        pickle.dump(train_set, open('data/streg/train.bin', 'wb'))
        pickle.dump(val_set, open('data/streg/val.bin', 'wb'))
        pickle.dump(testi_set, open('data/streg/testi.bin', 'wb'))
        pickle.dump(teste_set, open('data/streg/teste.bin', 'wb'))
        pickle.dump(vocab, open('data/streg/vocab.freq%d.bin'%(vocab_freq_cutoff), 'wb'))
        
    @staticmethod
    def load_map_file(filename):
        with open(filename) as f:
            lines = f.readlines()
        lines = [x.rstrip() for x in lines]
        maps = []
        for l in lines:
            fields = l.split(" ")
            num = int(fields[0])
            fields = fields[1:]
            if num == 0:
                maps.append([])
                continue
            m = []
            for f in fields:
                pair = f.split(",", 1)
                m.append((pair[0], pair[1]))
            maps.append(m)
        return maps

    @staticmethod
    def load_examples(filename):
        lines = _read_lines(filename)
        lines = [x.split(" ") for x in lines]
        lines = [[(y.split(",", 1)[0], y.split(",", 1)[1])
                    for y in x] for x in lines]
        return lines

    @staticmethod
    def load_rec(filename):
        with open(filename, "rb") as f:
            rec = pickle.load(f)
        return rec

    @staticmethod
    def load_regex_dataset(transition_system, split):
        prefix = 'data/streg/'
        src_file = join(prefix, "src-{}.txt".format(split))
        spec_file = join(prefix, "targ-{}.txt".format(split))
        map_file = join(prefix, "map-{}.txt".format(split))
        exs_file = join(prefix, "exs-{}.txt".format(split))
        rec_file = join(prefix, "rec-{}.pkl".format(split))

        exs_info = StReg.load_examples(exs_file)
        map_info = StReg.load_map_file(map_file)
        rec_info = StReg.load_rec(rec_file)

        examples = []
        for idx, (src_line, spec_line, str_exs, cmap, rec) in enumerate(zip(open(src_file), open(spec_file), exs_info, map_info, rec_info)):
            print(idx)
            
            src_line = src_line.rstrip()
            spec_line = spec_line.rstrip()
            src_toks = src_line.split()
            
            spec_toks = spec_line.rstrip().split()
            spec_ast = streg_expr_to_ast(transition_system.grammar, spec_toks)
            # sanity check
            reconstructed_expr = transition_system.ast_to_surface_code(spec_ast)
            # print("Spec", spec_line)
            # print("Rcon", reconstructed_expr)
            assert spec_line == reconstructed_expr

            tgt_actions = transition_system.get_actions(spec_ast)
            # sanity check
            hyp = Hypothesis()
            for action in tgt_actions:
                assert action.__class__ in transition_system.get_valid_continuation_types(hyp)
                if isinstance(action, ApplyRuleAction):
                    assert action.production in transition_system.get_valid_continuating_productions(hyp)
                hyp = hyp.clone_and_apply_action(action)

            assert hyp.frontier_node is None and hyp.frontier_field is None
            assert is_equal_ast(hyp.tree, spec_ast)

            expr_from_hyp = transition_system.ast_to_surface_code(hyp.tree)
            assert expr_from_hyp == spec_line

            tgt_action_infos = get_action_infos(src_toks, tgt_actions)
            
            example = Example(idx=idx,
                            src_sent=src_toks,
                            tgt_actions=tgt_action_infos,
                            tgt_code=spec_line,
                            tgt_ast=spec_ast,
                            meta={"str_exs": str_exs,
                                "const_map": cmap,
                                "worker_info": rec})
            examples.append(example)
        return examples


if __name__ == '__main__':
    StReg.process_regex_dataset()
