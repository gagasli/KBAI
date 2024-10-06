from itertools import combinations
class MonsterDiagnosisAgent:
    def __init__(self):
        # If you want to do any initial processing, add it here.
        pass

    def solve(self, diseases, patient):
        # Add your code here!
        #
        # The first parameter to this method is a list of diseases, represented as a
        # list of 2-tuples. The first item in each 2-tuple is the name of a disease. The
        # second item in each 2-tuple is a dictionary of symptoms of that disease, where
        # the keys are letters representing vitamin names ("A" through "Z") and the values
        # are "+" (for elevated), "-" (for reduced), or "0" (for normal).
        #
        # The second parameter to this method is a particular patient's symptoms, again
        # represented as a dictionary where the keys are letters and the values are
        # "+", "-", or "0".
        #
        # This method should return a list of names of diseases that together explain the
        # observed symptoms. If multiple lists of diseases can explain the symptoms, you
        # should return the smallest list. If multiple smallest lists are possible, you
        # may return any sufficiently explanatory list.
        self.diseases = diseases

        v_list = [chr(id) for id in range(ord('A'), ord('Z') + 1)]
        self.v_list_2 = []

        for v_id in v_list:
            value = None
            for _, d_dict in diseases.items():
                if not value: value = d_dict[v_id]
                elif d_dict[v_id] != value: 
                    self.v_list_2.append(v_id)
                    break

        for d_num in range(1, len(diseases)+1):
            combos = list(combinations(diseases.keys(), d_num))
            for combo in combos:
                sym = self.get_sym(combo)
                if self.find_sym(sym, patient): return list(combo)
                

    def get_sym(self, d_tuple):
        sym_dict = {}
        for v_id in self.v_list_2:
            value = sum(string2num(self.diseases[d][v_id]) for d in d_tuple)
            sym_dict[v_id] = num2string(value)
        
        return sym_dict

    def find_sym(self, sym, patient):
        for v_id in self.v_list_2:
            if sym[v_id] != patient[v_id]: return False
        return True

def string2num(string):
    if string == "+": return 1
    if string == "-": return -1
    return 0

def num2string(num):
    if num > 0: return "+"
    if num < 0: return "-"
    return "0"
