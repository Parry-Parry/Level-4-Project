import tensorflow
import transformers

print("building...")
model = transformers.TFEncoderDecoderModel.from_encoder_decoder_pretrained("distilroberta-base", "distilroberta-base")
print("saving...")
model.save_pretrained("smallbert")

