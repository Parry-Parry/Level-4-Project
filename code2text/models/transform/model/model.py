from transformers import RobertaConfig, RobertaModel, RobertaTokenizerFast


tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
configuration = RobertaConfig()
model = RobertaModel(configuration)

configuration = model.config