import pickle

class DecodeResult:
    def __init__(self, programs, num_exec, eval_result=None):
        self.decodes = programs
        self.num_exec = num_exec
        self.eval_result = eval_result