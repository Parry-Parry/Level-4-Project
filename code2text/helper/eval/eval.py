# Take output and ground

# Tokenize
# Calculate BLEU | METEOR | ROUGE

from models.eval.metrics.rouge import rouge
from models.eval.metrics.bleu import compute_bleu
from nltk.translate import single_meteor_score



class Evaluator():
    metrics = {}
    def __init__(self) -> None:
        pass


    def _bleu(self, out_seq, ground_seq):
        return compute_bleu(ground_seq, out_seq)


    def _meteor(self, out_seq, ground_seq):
        N = len(out_seq)
        score = 0
        for ground, out in zip(ground_seq, out_seq):
            score += single_meteor_score(ground, out)**(-1)
        return N / score

    def _rouge(self, out_seq, ground_seq):
        return rouge(out_seq, ground_seq)  

    def evaluate(self, out, ground, path=None):
        # tokenize seqs
        self.metrics['bleu'] = self._bleu(out, ground)
        self.metrics['meteor'] = self._meteor(out, ground)
        self.metrics['rouge'] = self._rouge(out, ground)

        if path:
            #pickle and write to path
            pass