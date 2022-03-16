import tensorflow
import transformers
from transformers import RobertaConfig, TFRobertaModel

h=8
l=3
hs=384

print("building...")
configuration = RobertaConfig(num_attention_heads=h, num_hidden_layers=l, hidden_size=hs)
model = TFRobertaModel(configuration)
model.save_pretrained("tmp_roberta")

model = transformers.TFEncoderDecoderModel.from_encoder_decoder_pretrained("tmp_roberta", "tmp_roberta")
print("saving...")
model.save_pretrained("medsmallbert")

