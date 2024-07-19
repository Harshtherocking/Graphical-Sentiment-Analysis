import torch

class DepTokenizer():
    def __init__ (self):
        self.token_dict = {
                        'acl': 0,
                        'acomp': 1,
                        'advcl': 2,
                        'advmod': 3,
                        'agent': 4,
                        'amod': 5,
                        'appos': 6,
                        'attr': 7,
                        'aux': 8,
                        'auxpass': 9,
                        'case': 10,
                        'cc': 11,
                        'ccomp': 12,
                        'compound': 13,
                        'conj': 14,
                        'csubj': 15,
                        'csubjpass': 16,
                        'dative': 17,
                        'dep': 18,
                        'det': 19,
                        'dobj': 20,
                        'expl': 21,
                        'intj': 22,
                        'mark': 23,
                        'meta': 24,
                        'neg': 25,
                        'nounmod': 26,
                        'npmod': 27,
                        'nsubj': 28,
                        'nsubjpass': 29,
                        'nummod': 30,
                        'oprd': 31,
                        'parataxis': 32,
                        'pcomp': 33,
                        'pobj': 34,
                        'poss': 35,
                        'preconj': 36,
                        'predet': 37,
                        'prep': 38,
                        'prt': 39,
                        'punct': 40,
                        'quantmod': 41,
                        'relcl': 42,
                        'root': 43,
                        'xcomp': 44};

        self.num_dep = len(self.token_dict)
        return None;

    def __call__ (self, dep : str | list[str])  -> torch.Tensor: 
        if isinstance(dep,str):
            return torch.tensor(self.token_dict[dep])
        else:
            return torch.tensor([self.token_dict[t] for t in dep])




if __name__ == "__main__":
    tokenizer = DepTokenizer()
    __import__('pprint').pprint(tokenizer("pobj"))
    __import__('pprint').pprint(tokenizer(["pobj", "root", "oprd"]))
    
