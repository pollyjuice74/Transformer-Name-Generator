import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures

# read in all the words
words = open('names.txt', 'r').read().splitlines()
#words[:8]

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
#print(itos)

block_size = 3 # context length: how many characters do we take to predict the next one?

'''
X, Y = [], [] # X list of context, Y list of labels(predictions)
for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]

X = torch.tensor(X) # (32, 3) tensor of all contexts
Y = torch.tensor(Y) # (..., 1) tensor of labels
'''

def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27, 10), generator=g) # Make tensor with random weights
W1 = torch.randn((30, 300), generator=g) # 3*C[,j] dimensional imbedding and arbitrary variable
b1 = torch.randn(300, generator=g)
# Second layer 
W2 = torch.randn((300, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

for p in parameters:
    p.requires_grad = True

lre = torch.linspace(-3, 0, 1000) # Learning rate exponent, between 0.001 and 1
lrs = 10**lre

lri = []
lossi = []
stepi = []
for i in range(100000):
    # minibatches
    ix = torch.randint(0, Xtr.shape[0], (32,))
    
    ###FORWARD PASS
    emb = C[Xtr[ix]] # Embeding, index into C with X, creates a (32, 3, 2)
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # Manipulate emb to be (32, 6) and then you can multiply and sum
    logits = h @ W2 + b2 # (32, 27)
    #torch.cat([emb[:, 0, :], emb[:, 1, :], emb[:, 2, :]], 1)
    #counts = logits.exp()
    #prob = counts / counts.sum(1, keepdims=True)
    #loss = -prob[torch.arange(32), Y].log().mean()
    loss = F.cross_entropy(logits, Ytr[ix]) # More efficient numerically and in computatioinal terms than the previous implementation
    ###BACKWARD PASS
    for p in parameters:
        p.grad = None
    loss.backward()
    # update
    #lr = lrs[i] # Learning rate
    lr = 0.1 if i < 100000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
    print(loss.item())
        # track stats
    stepi.append(i)
    lossi.append(loss.log10().item())
plt.plot(stepi, lossi)

# training split, dev/validation split, test split
# 80%, 10%, 10%

###SAMPLING 
for _ in range(200):
    out = []
    context = [0] * block_size
    while True: 
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        out.append(ix)
        if ix == 0:
            break 

    print(''.join(itos[i] for i in out))