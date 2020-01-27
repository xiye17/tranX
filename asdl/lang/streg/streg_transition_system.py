# coding=utf-8
from .streg_utils import * 
import ast

from asdl.transition_system import TransitionSystem, GenTokenAction

try:
    from cStringIO import StringIO
except:
    from io import StringIO
from asdl.asdl import *
from asdl.asdl_ast import RealizedField, AbstractSyntaxTree
from common.registerable import Registrable
"""
# define primitive fields
int, cc, tok

regex = Not(regex arg)
    | Star(regex arg)
    | Concat(regex left, regex right)
    | Or(regex left, regex right)
    | And(regex left, regex right)
    | StartWith(regex arg)
    | EndWith(regex arg)
    | Contain(regex arg)
    | RepAtleast(regex arg, int k)
    | RepeatAtleast(regex arg, int k)
    | RepeatRange(reg arg, int k1, int k2)
    | CharClass(cc arg)
    | Token(tok arg) #  <x> single tokn
    | ConstSym(csymbl arg) # const0 const1
    | String(str arg)  # string const(<str>)
"""

_NODE_CLASS_TO_RULE = {
    "not": "Not",
    "notcc": "NotCC",
    "star": "Star",
    "optional": "Optional",
    "startwith": "StartWith",
    "endwith": "EndWith",
    "contain": "Contain",
    "concat": "Concat",
    "and": "And",
    "or": "Or",
    "repeat": "Repeat",
    "repeatatleast": "RepeatAtleast",
    "repeatrange": "RepeatRange",
    "const": "String"
}

def streg_ast_to_asdl_ast(grammar, reg_ast):
    if reg_ast.children:
        rule = _NODE_CLASS_TO_RULE[reg_ast.node_class]
        prod = grammar.get_prod_by_ctr_name(rule)
        # unary
        if rule in ["Not", "Star", "StartWith", "EndWith", "Contain", "NotCC", "Optional"]:
            child_ast_node = streg_ast_to_asdl_ast(grammar, reg_ast.children[0])
            ast_node = AbstractSyntaxTree(prod,
                                            [RealizedField(prod['arg'], child_ast_node)])
            return ast_node
        elif rule in ["Concat", "And", "Or"]:
            left_ast_node = streg_ast_to_asdl_ast(grammar, reg_ast.children[0])
            right_ast_node = streg_ast_to_asdl_ast(grammar, reg_ast.children[1])
            ast_node = AbstractSyntaxTree(prod,
                                            [RealizedField(prod['left'], left_ast_node),
                                            RealizedField(prod['right'], right_ast_node)])
            return ast_node
        elif rule in ["RepeatAtleast", "Repeat"]:
            # primitive node
            # RealizedField(prod['predicate'], value=node_name)
            child_ast_node = streg_ast_to_asdl_ast(grammar, reg_ast.children[0])
            int_real_node = RealizedField(prod['k'], str(reg_ast.params[0]))
            ast_node = AbstractSyntaxTree(prod, [RealizedField(prod['arg'], child_ast_node), int_real_node])
            return ast_node
        elif rule in ["RepeatRange"]:
            child_ast_node = streg_ast_to_asdl_ast(grammar, reg_ast.children[0])
            int_real_node1 = RealizedField(prod['k1'], str(reg_ast.params[0]))
            int_real_node2 = RealizedField(prod['k2'], str(reg_ast.params[1]))
            ast_node = AbstractSyntaxTree(prod, [RealizedField(prod['arg'], child_ast_node), int_real_node1, int_real_node2])
            return ast_node
        elif rule in ["String"]:
            return AbstractSyntaxTree(prod, [RealizedField(prod['arg'], reg_ast.children[0].node_class)])
        else:
            raise ValueError("wrong node class", reg_ast.node_class)
    else:
        if reg_ast.node_class in ["<num>", "<let>", "<spec>", "<low>", "<cap>", "<any>"]:
            rule = "CharClass"
        elif reg_ast.node_class.startswith("const") and reg_ast.node_class[5:].isdigit():
            rule = "ConstSym"
        elif reg_ast.node_class.startswith("<") and reg_ast.node_class.endswith(">"):
            rule = "Token"
        else:
            raise ValueError("wrong node class", reg_ast.node_class)
        prod = grammar.get_prod_by_ctr_name(rule)
        return AbstractSyntaxTree(prod, [RealizedField(prod['arg'], reg_ast.node_class)])

def streg_expr_to_ast(grammar, reg_tokens):
    reg_ast = build_streg_ast_from_toks(reg_tokens, 0)[0]
    assert reg_ast.tokenized_logical_form() == reg_tokens
    return streg_ast_to_asdl_ast(grammar, reg_ast)

def asdl_ast_to_streg_ast(asdl_ast):
    rule = asdl_ast.production.constructor.name
    if rule in ["CharClass", "ConstSym", "Token"]:
        return StRegNode(asdl_ast['arg'].value)
    elif rule in ["String"]:
        c_val = asdl_ast['arg'].value
        if c_val.startswith("<") and c_val.endswith(">"):
            child = StRegNode(c_val)
            return StRegNode("const", [child])
        else:
            return StRegNode("none")
    elif rule in ["Not", "Star", "StartWith", "EndWith", "Contain", "Concat", "And", "Or", "NotCC", "Optional"]:
        node_class = rule.lower()
        return StRegNode(node_class, [asdl_ast_to_streg_ast(x.value) for x in asdl_ast.fields])
    elif rule in ["RepeatAtleast", "Repeat"]:
        node_class = rule.lower()
        if asdl_ast['k'].value.isdigit():
            param = int(asdl_ast['k'].value)
            child_node = asdl_ast_to_streg_ast(asdl_ast['arg'].value)
            return StRegNode(node_class, [child_node], [param])
        else:
            return StRegNode("none")
    elif rule in ["RepeatRange"]:
        node_class = rule.lower()
        if asdl_ast['k1'].value.isdigit() and asdl_ast['k2'].value.isdigit():
            params = [int(asdl_ast['k1'].value), int(asdl_ast['k2'].value)]
            child_node = asdl_ast_to_streg_ast(asdl_ast['arg'].value)
            return StRegNode(node_class, [child_node], params)
        else:
            return StRegNode("none")
    else:
        raise ValueError("wrong ast rule", rule)

def asdl_ast_to_streg_expr(asdl_ast):
    reg_ast = asdl_ast_to_streg_ast(asdl_ast)
    return " ".join(reg_ast.tokenized_logical_form())

# neglet created time
def is_equal_ast(this_ast, other_ast):
    if not isinstance(other_ast, this_ast.__class__):
        return False
    # print(this_ast, other_ast)

    if isinstance(this_ast, AbstractSyntaxTree):
        if this_ast.production != other_ast.production:
            return False

        if len(this_ast.fields) != len(other_ast.fields):
            return False
        for this_f, other_f in zip(this_ast.fields, other_ast.fields):
            if not is_equal_ast(this_f.value, other_f.value):
                return False
        return True
    else:
        return this_ast == other_ast

@Registrable.register('streg')
class StRegTransitionSystem(TransitionSystem):
    def compare_ast(self, hyp_ast, ref_ast):
        return is_equal_ast(hyp_ast, ref_ast)

    def ast_to_surface_code(self, asdl_ast):
        return asdl_ast_to_streg_expr(asdl_ast)
    
    def surface_code_to_ast(self, code):
        return streg_expr_to_ast(self.grammar, code)
    
    def hyp_correct(self, hype, example):
        return is_equal_ast(hype.tree, example.tgt_ast)
    
    def tokenize_code(self, code, mode):
        return code.split()
    
    def get_primitive_field_actions(self, realized_field):
        assert realized_field.cardinality == 'single'
        if realized_field.value is not None:
            return [GenTokenAction(realized_field.value)]
        else:
            return []
