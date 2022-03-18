import transformers


tmp = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained("nyu-mll/roberta-med-small-1M-1", "nyu-mll/roberta-med-small-1M-1")
print("saving...")
model.save_pretrained("model")

