from transformers import TFEncoderDecoderModel

print("building...")
model = TFEncoderDecoderModel.from_encoder_decoder_pretrained("distilroberta-base", "distilroberta-base")
print("saving...")
model.save_pretrained("/users/level4/2393265p/workspace/l4project/code/smallbert")

