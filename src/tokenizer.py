


class ScratchyTokenizer:
    def __init__(self, chars: str):
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch:i for i, ch in enumerate(self.chars)}
        self.itos = {i:ch for i, ch in enumerate(self.chars)}

    def encode(self, string):
        return [self.stoi[c] for c in string]

    def decode(self, ids):
        return ''.join([self.itos[i] for i in ids])
