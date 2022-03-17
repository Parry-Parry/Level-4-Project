import transformers

print("building...")
model = transformers.EncoderDecoderModel.from_encoder_decoder_pretrained("sshleifer/tiny-distilroberta-base", "sshleifer/tiny-distilroberta-base")
print("saving...")
model.save_pretrained("model")

