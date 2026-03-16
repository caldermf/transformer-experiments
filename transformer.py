import torch
import torch.nn as nn
import torch.nn.functional as F

###
block_size = 8
batch_size = 32
hidden_1 = 32
###

torch.manual_seed(217)

with open('transcript.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

itos = {i : chars[i] for i in range(len(chars))}
stoi = {chars[i] : i for i in range(len(chars))}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = len(data)
train_data = data[:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

def get_batch(mode):
    if mode == 'val':
        curr_data = val_data
    elif mode == 'test':
        curr_data = test_data
    else:
        curr_data = train_data
    data_len = len(curr_data)
    block_starts = torch.randint(data_len - block_size - 1, (batch_size,))
    X = torch.stack([curr_data[start:start+block_size] for start in block_starts])
    Y = torch.stack([curr_data[start+1:start+1+block_size] for start in block_starts])
    return X, Y

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weights = torch.randn((num_embeddings, embedding_dim)) / (embedding_dim ** 0.5)

    def __call__(self, to_embed):
        # to_embed is a tensor of shape (batch_size, block_size) with dtype=torch.long
        # returns tensor of shape (batch_size, block_size, embedding_dim)
        return self.weights[to_embed]

    def parameters(self):
        return [self.weights]

class Linear:
    def __init__(self, dim_in, dim_out, have_bias=True):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weights = torch.randn((dim_in, dim_out)) / (dim_in ** 0.5)
        self.have_bias = have_bias
        self.bias = torch.zeros(dim_out)
    
    def __call__(self, vect_in):
        a = vect_in @ self.weights
        if self.have_bias:
            a += self.bias
        return a

    def parameters(self):
        if self.have_bias:
            return [self.weights, self.bias]
        else:
            return [self.weights]
        
class Activation:
    def __call__(self, x):
        return torch.tanh(x)
    
    def parameters(self):
        return []

class Sequential:
    def __init__(self, layers):
        self.layers = layers
    
    def __call__(self, x):
        for l in self.layers:
            x = l(x)
        return x
    
    def parameters(self):
        p = []
        for l in self.layers:
            p.extend(l.parameters())
        return p

model = Sequential([Embedding(vocab_size, hidden_1),
                    Linear(hidden_1, hidden_1),
                    Activation(),
                    Linear(hidden_1, vocab_size)])

parameters = model.parameters()
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True

max_steps = 80000
lr = 0.1
lossi = []
for i in range(max_steps):
    X, Y = get_batch('train') #X and Y are (B, T)

    #forward pass
    logits = model(X) #(B, T, vocab_size)
    loss = F.cross_entropy(logits.view(-1, vocab_size), Y.view(-1))

    #backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    for p in parameters:
        p.data += -lr * p.grad
    
    if i % 10000 == 0:
        print(f'{i:7d}/{max_steps:7d}: {loss.item():4f}')
    lossi.append(loss.log10().item())
    
def generate(model, start_text, max_new_tokens):
    model_input = torch.tensor([encode(start_text)], dtype=torch.long) #(1, start_text_length)
    for _ in range(max_new_tokens):
        x = model_input[:, -block_size:] # (1, T)
        logits = model(x) # (1, T, vocab_size)
        last_logits = logits[:,-1,:] # (1, vocab_size), logits for last character
        probs = F.softmax(last_logits, dim=1) #(1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1) # (1, vocab_size)
        model_input = torch.cat([model_input, next_token], dim=1)
    return decode(model_input[0].tolist())

print(generate(model, "HELLO AT ALL, MORTON HERE! I just wanted to say", 300))
print(vocab_size)


