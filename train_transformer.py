from easy_nlp.model import Transformer
import keras


num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

# x = keras.Input(shape=(10, 128), batch_size=2)
# attn = GlobalSelfAttention(num_heads, d_model)
# y = attn(x)
# m = keras.Model(x, y)
# print(m.summary())

x1 = keras.Input(shape=(20,), batch_size=32)
x2 = keras.Input(shape=(20,), batch_size=32)
transformer = Transformer(
    n_layer=num_layers,
    d_model=d_model,
    n_head=num_heads,
    d_ff=dff,
    vocab_size=30000,
    dropout=dropout_rate)
y = transformer((x1, x2))
model = keras.Model([x1, x2], y)

print(model.summary())