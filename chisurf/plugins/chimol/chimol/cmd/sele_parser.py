from __future__ import annotations

import re
from typing import List, Optional, Union, Callable
from dataclasses import dataclass

import numpy as np


# Tokenizer

@dataclass
class Token:
    type: str
    value: str
    start: int
    end: int

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


TOKEN_TYPES = [
    ("SPACE", r"\s+"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("AND", r"(?i)\band\b|&"),
    ("OR", r"(?i)\bor\b|\|"),
    ("NOT", r"(?i)\bnot\b|!"),
    ("WITHIN", r"(?i)\bwithin\b"),
    ("OF", r"(?i)\bof\b"),
    ("AROUND", r"(?i)\baround\b"),
    ("EXPAND", r"(?i)\bexpand\b"),
    ("BYRES", r"(?i)\bbyres\b"),
    ("BYMOL", r"(?i)\bbymol\b"),
    ("BYOBJ", r"(?i)\bbyobj\b"),
    ("TO", r"(?i)\bto\b"),
    ("ALL", r"(?i)\ball\b|\*"),
    ("NONE", r"(?i)\bnone\b"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("FLOAT", r"\d+\.\d+"),
    ("INT", r"\d+"),
    ("COLON", r":"),
    ("SLASH", r"/"),
]

TOKEN_REGEX = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_TYPES))


def tokenize(expr: str) -> List[Token]:
    tokens = []
    for match in TOKEN_REGEX.finditer(expr):
        kind = match.lastgroup
        value = match.group()
        if kind == "SPACE":
            continue
        tokens.append(Token(kind, value, match.start(), match.end()))
    return tokens


# AST Nodes

class ASTNode:
    pass

@dataclass
class AllNode(ASTNode):
    pass

@dataclass
class NoneNode(ASTNode):
    pass

@dataclass
class IdentNode(ASTNode):
    value: str

@dataclass
class ValueNode(ASTNode):
    value: Union[int, float, str]

@dataclass
class UnaryOpNode(ASTNode):
    op: str
    expr: ASTNode

@dataclass
class BinaryOpNode(ASTNode):
    left: ASTNode
    op: str
    right: ASTNode

@dataclass
class ListNode(ASTNode):
    items: List[ASTNode]

@dataclass
class RangeNode(ASTNode):
    start: ASTNode
    end: ASTNode

@dataclass
class PropertyOpNode(ASTNode):
    prop: str
    values: ASTNode

@dataclass
class PrefixSelectNode(ASTNode):
    prefix: str
    values: ASTNode

@dataclass
class DistanceOpNode(ASTNode):
    op: str
    dist: float
    target: ASTNode

@dataclass
class MacroNode(ASTNode):
    obj: str
    chain: str
    resi: str
    name: str

# Parser

class ParserError(Exception):
    pass

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            token = self.tokens[self.pos]
            self.pos += 1
            return token
        return None
        
    def match(self, *expected_types) -> Optional[Token]:
        token = self.peek()
        if token and token.type in expected_types:
            return self.advance()
        return None

    def expect(self, expected_type) -> Token:
        token = self.advance()
        if not token or token.type != expected_type:
            raise ParserError(f"Expected {expected_type}, got {token.type if token else 'EOF'}")
        return token

    def parse(self) -> ASTNode:
        if not self.tokens:
            return AllNode()
        node = self.parse_expr()
        if self.pos < len(self.tokens):
            raise ParserError(f"Unexpected token {self.peek()} at end of expression")
        return node

    def parse_expr(self) -> ASTNode:
        return self.parse_or()

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match("OR"):
            right = self.parse_and()
            left = BinaryOpNode(left, "OR", right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while True:
            # explicit 'and' or implicit 'and' (space adjacent)
            if self.match("AND"):
                right = self.parse_not()
                left = BinaryOpNode(left, "AND", right)
            else:
                # check if the next token can start a secondary expression
                token = self.peek()
                if token and token.type in ("IDENT", "LPAREN", "NOT", "ALL", "NONE", "BYRES", "BYMOL", "BYOBJ", "AROUND", "EXPAND"):
                    right = self.parse_not()
                    left = BinaryOpNode(left, "AND", right)
                else:
                    break
        return left

    def parse_not(self) -> ASTNode:
        if self.match("NOT"):
            expr = self.parse_not()
            return UnaryOpNode("NOT", expr)
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        token = self.peek()
        if not token:
            raise ParserError("Unexpected end of expression")

        if self.match("LPAREN"):
            node = self.parse_expr()
            self.expect("RPAREN")
            return node

        if self.match("ALL"):
            return AllNode()
        
        if self.match("NONE"):
            return NoneNode()

        if token.type in ("IDENT"):
            ident = self.advance().value
            lower = ident.lower()
            
            # Distance ops
            if lower in ("around", "expand"):
                val = self.advance()
                if val.type not in ("FLOAT", "INT"):
                    raise ParserError("Expected numeric distance")
                dist = float(val.value)
                target = self.parse_primary()
                return DistanceOpNode(lower, dist, target)
                
            # By ops
            if lower in ("byres", "bymol", "byobj"):
                target = self.parse_primary()
                return UnaryOpNode(lower, target)

            property_names = ("res", "resi", "residue", "name", "chain", "elem", "ss", "segi", "model", "object", "obj", "b", "q", "id")
            if lower in property_names:
                values = self.parse_value_list()
                return PropertyOpNode(lower, values)
            
            if lower == "within":
                val = self.advance()
                if val.type not in ("FLOAT", "INT"):
                    raise ParserError("Expected numeric distance for within")
                dist = float(val.value)
                self.expect("OF")
                target = self.parse_primary()
                return DistanceOpNode("within", dist, target)

            # Property prefix
            prefixes = ("c.", "m.", "o.", "s.", "r.", "n.", "i.", "e.")
            for pre in prefixes:
                if lower.startswith(pre):
                    prefix = pre[0]
                    value_str = ident[2:]
                    # Need to parse further value list if present but start with this
                    values = None # self.parse_value_list() # TODO
                    return PrefixSelectNode(prefix, ValueNode(value_str))
            
            return IdentNode(ident)
        
        # Macro syntax /obj/chain/res/name
        if self.match("SLASH"):
            return self.parse_macro()
            
        raise ParserError(f"Unexpected token {token}")

    def parse_value_list(self) -> ASTNode:
        items = []
        items.append(self.parse_value_range())
        while self.match("PLUS"):
            items.append(self.parse_value_range())
        if len(items) == 1:
            return items[0]
        return ListNode(items)

    def parse_value_range(self) -> ASTNode:
        start = self.parse_value()
        is_range = False
        if self.match("MINUS") or self.match("TO"):
             end = self.parse_value()
             return RangeNode(start, end)
        if self.match("COLON"):
             end = self.parse_value()
             return RangeNode(start, end)
        return start

    def parse_value(self) -> ASTNode:
        if self.match("MINUS"):
             token = self.advance()
             if not token: raise ParserError("Unexpected end of expression after '-'")
             if token.type == "INT":
                  return ValueNode(-int(token.value))
             elif token.type == "FLOAT":
                  return ValueNode(-float(token.value))
             raise ParserError(f"Expected number after '-' but got {token.type}")
             
        token = self.advance()
        if not token: raise ParserError("Unexpected end of expression")
        if token.type in ("INT", "FLOAT"):
             return ValueNode(float(token.value) if token.type == "FLOAT" else int(token.value))
        if token.type == "IDENT":
             return ValueNode(token.value)
        raise ParserError(f"Expected value, got {token.type}")

    def parse_macro(self) -> ASTNode:
        # Simplified macro parsing obj/chain/res/name
        parts = ["", "", "", ""]
        part_idx = 0
        while part_idx < 4:
            token = self.peek()
            if not token: break
            if token.type == "SLASH":
                self.advance()
                part_idx += 1
                continue
            
            # accumulate until slash
            val = ""
            while True:
                t = self.peek()
                if not t or t.type == "SLASH" or t.type == "SPACE" or t.type == "RPAREN":
                    break
                val += self.advance().value
            parts[part_idx] = val
        return MacroNode(obj=parts[0], chain=parts[1], resi=parts[2], name=parts[3])

# Evaluator

class Evaluator:
    def __init__(self, viewer, default_object_id=None):
        self.viewer = viewer
        self.default_object_id = default_object_id
        
    def evaluate(self, expr: str, object_id=None) -> np.ndarray:
        tokens = tokenize(expr)
        parser = Parser(tokens)
        ast = parser.parse()
        obj_id = object_id or self.default_object_id
        return self.eval_node(ast, obj_id)
        
    def eval_node(self, node: ASTNode, object_id: str) -> np.ndarray:
        if isinstance(node, AllNode):
            return self._get_all_mask(object_id)
        elif isinstance(node, NoneNode):
            return self._get_none_mask(object_id)
        elif isinstance(node, IdentNode):
            # Try to resolve named selection or object name
            if node.value.lower() == "polymer":
                 return self._get_polymer_mask(object_id)
            if node.value.lower() == "solvent":
                 return self._get_solvent_mask(object_id)
            return self._get_ident_mask(node.value, object_id)
        elif isinstance(node, UnaryOpNode):
            if node.op == "NOT":
                return ~self.eval_node(node.expr, object_id)
            elif node.op == "byres":
                mask = self.eval_node(node.expr, object_id)
                return self._expand_to_residues(mask, object_id)
            elif node.op == "bymol":
                mask = self.eval_node(node.expr, object_id)
                return self._expand_to_molecules(mask, object_id)
        elif isinstance(node, BinaryOpNode):
            left = self.eval_node(node.left, object_id)
            right = self.eval_node(node.right, object_id)
            if node.op == "AND":
                return left & right
            elif node.op == "OR":
                return left | right
        elif isinstance(node, PropertyOpNode):
            return self._eval_property(node.prop, node.values, object_id)
        elif isinstance(node, PrefixSelectNode):
             # map 'c' -> chain, 'r' -> resi, etc.
             prop_map = {'c': 'chain', 'r': 'resi', 'n': 'name', 'e': 'elem', 'i': 'id', 's': 'segi', 'o': 'obj'}
             prop = prop_map.get(node.prefix, "unknown")
             return self._eval_property(prop, node.values, object_id)
        elif isinstance(node, DistanceOpNode):
             return self._eval_distance(node.op, node.dist, node.target, object_id)
        elif isinstance(node, MacroNode):
             return self._eval_macro(node, object_id)
             
        raise NotImplementedError(f"Evaluation of {type(node)} not implemented")

    def _get_all_mask(self, object_id: str) -> np.ndarray:
        try:
             entry = self.viewer._objects.get(object_id)
             n_atoms = entry.state.atoms.shape[0]
             return np.ones(n_atoms, dtype=bool)
        except Exception:
             return np.zeros(0, dtype=bool)

    def _get_none_mask(self, object_id: str) -> np.ndarray:
        try:
             entry = self.viewer._objects.get(object_id)
             n_atoms = entry.state.atoms.shape[0]
             return np.zeros(n_atoms, dtype=bool)
        except Exception:
             return np.zeros(0, dtype=bool)
             
    def _get_polymer_mask(self, object_id: str) -> np.ndarray:
        # Heuristic: mostly proteins/nucleic acids
        # For now, return all since we don't have polymer flags parsed
        return self._get_all_mask(object_id)

    def _get_solvent_mask(self, object_id: str) -> np.ndarray:
        # Heuristic: HOH, WAT
        return self._eval_property("resn", ValueNode("HOH"), object_id) | self._eval_property("resn", ValueNode("WAT"), object_id)

    def _get_ident_mask(self, name: str, object_id: str) -> np.ndarray:
         # Could be named selection or object name fallback
         # Return none for now if it's not handled
         return self._get_none_mask(object_id)

    def _eval_property(self, prop: str, values_node: ASTNode, object_id: str) -> np.ndarray:
        none_mask = self._get_none_mask(object_id)
        if len(none_mask) == 0: return none_mask
        
        try:
            entry = self.viewer._objects.get(object_id)
            state = entry.state
            atoms = getattr(state, "atoms", None)
            
            # Map property name to array/column
            prop = prop.lower()
            
            if prop in ("name", "atom"):
                 if atoms is None: return none_mask
                 arr = np.char.strip(atoms["atom_name"].astype(str)).astype(object)
                 arr_lower = np.char.lower(arr.astype(str))
            elif prop in ("res", "resi", "residue", "id"):
                 res_ids = state.all_atom_res_ids
                 if res_ids is None: return none_mask
                 arr_lower = np.char.lower(res_ids.astype(str))
            elif prop in ("resn", "resname"):
                 # Residue names from atoms if available
                 if atoms is not None and "res_name" in atoms.dtype.fields:
                      arr_lower = np.char.lower(np.char.strip(atoms["res_name"].astype(str)))
                 elif state.residue_names is not None and state.all_atom_res_ids is not None:
                      # Map residue index to name
                      # This assumes all_atom_res_ids are INDICES into residue_names
                      # In ChiMol, they often are globally unique IDs or indices.
                      # Let's assume indices for now as a fallback.
                      res_indices = state.all_atom_res_ids
                      arr_lower = np.char.lower(state.residue_names[res_indices].astype(str))
                 else:
                      return none_mask
            elif prop in ("chain", "c"):
                 # Check atoms first
                 if atoms is not None and "chain" in atoms.dtype.fields:
                      arr_lower = np.char.lower(np.char.strip(atoms["chain"].astype(str)))
                 elif state.residue_chain_ids is not None and state.all_atom_res_ids is not None:
                      res_indices = state.all_atom_res_ids
                      # residue_chain_ids is per-residue. Map to atoms.
                      # Again assuming all_atom_res_ids are residue indices.
                      arr_lower = np.char.lower(state.residue_chain_ids[res_indices].astype(str))
                 else:
                      return none_mask
            elif prop in ("elem", "e"):
                 if atoms is not None and "element" in atoms.dtype.fields:
                      arr = np.char.strip(atoms["element"].astype(str))
                      arr_lower = np.char.lower(arr)
                 else:
                      # try to guess from atom name
                      if atoms is not None:
                           names = np.char.strip(atoms["atom_name"].astype(str))
                           # simple guess: first letter
                           arr_lower = np.char.lower(np.array([n[0] for n in names]))
                      else:
                           return none_mask
            else:
                 return none_mask
                 
            return self._match_values(arr_lower, values_node)
            
        except Exception:
            return none_mask
            
    def _match_values(self, arr: np.ndarray, values_node: ASTNode) -> np.ndarray:
         mask = np.zeros(arr.shape, dtype=bool)
         
         def _match_single(node: ASTNode):
             if isinstance(node, ValueNode):
                 val = str(node.value).lower()
                 mask[...] |= (arr == val)
             elif isinstance(node, RangeNode):
                 # Convert array to numeric if possible
                 try:
                     num_arr = arr.astype(float) # handle both ints and floats
                     start_val = float(node.start.value)
                     end_val = float(node.end.value)
                     if start_val > end_val:
                         start_val, end_val = end_val, start_val
                     mask[...] |= (num_arr >= start_val) & (num_arr <= end_val)
                 except Exception:
                     pass # Not numeric
             elif isinstance(node, ListNode):
                 for item in node.items:
                     _match_single(item)

         _match_single(values_node)
         return mask

    def _expand_to_residues(self, mask: np.ndarray, object_id: str) -> np.ndarray:
         if not np.any(mask): return mask
         try:
             entry = self.viewer._objects.get(object_id)
             res_ids = entry.state.all_atom_res_ids
             # find unique res_ids that have at least one True in mask
             selected_res_ids = np.unique(res_ids[mask])
             # return mask where res_id is in selected_res_ids
             return np.isin(res_ids, selected_res_ids)
         except Exception:
             return mask

    def _expand_to_molecules(self, mask: np.ndarray, object_id: str) -> np.ndarray:
         # Same as chains for now
         if not np.any(mask): return mask
         try:
             entry = self.viewer._objects.get(object_id)
             chain_ids = entry.state.all_atom_chain_ids
             selected_chain_ids = np.unique(chain_ids[mask])
             return np.isin(chain_ids, selected_chain_ids)
         except Exception:
             return mask

    def _eval_distance(self, op: str, dist: float, target_node: ASTNode, object_id: str) -> np.ndarray:
         target_mask = self.eval_node(target_node, object_id)
         if not np.any(target_mask):
             return self._get_none_mask(object_id)
             
         try:
             entry = self.viewer._objects.get(object_id)
             coords = entry.state.all_atom_coords
             
             target_coords = coords[target_mask]
             
             # cdist can be memory intensive, use broadcasting or KDTree if available
             from scipy.spatial import cKDTree
             tree = cKDTree(coords)
             
             res = tree.query_ball_point(target_coords, dist)
             
             mask = self._get_none_mask(object_id)
             for indices in res:
                  mask[indices] = True
                  
             if op == "within":
                  # within returns intersection, around returns neighborhood minus target
                  pass # Actually within usually filters another set
             elif op == "around":
                  mask &= ~target_mask
             # expand includes target
                  
             return mask
             
         except Exception:
             return self._get_none_mask(object_id)

    def _eval_macro(self, node: MacroNode, object_id: str) -> np.ndarray:
         mask = self._get_all_mask(object_id)
         if node.chain:
             mask &= self._eval_property("chain", ValueNode(node.chain), object_id)
         if node.resi:
             mask &= self._eval_property("resi", ValueNode(node.resi), object_id)
         if node.name:
             mask &= self._eval_property("name", ValueNode(node.name), object_id)
         return mask

