from transformers import RobertaTokenizer, TFRobertaForCausalLM
import tensorflow as tf

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

model = TFRobertaForCausalLM.from_pretrained("roberta-base")



