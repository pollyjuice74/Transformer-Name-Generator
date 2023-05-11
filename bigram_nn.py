import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().split()

chars = sorted(list(set(''.join(words))))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

# create traning set of bigrams (x, y)
    # xs are ix1
    # ys are ix2
xs, ys = [], []

for w in words:
    chrs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chrs, chrs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# create tensors
xs = torch.tensor(xs) # if ix1 then:
ys = torch.tensor(ys) # ix2 should have high prob
num = xs.nelement() # number of xs
#print(xs, ys)

g = torch.Generator().manual_seed(2147483647) # randomly initialize 27 neurons weights
W = torch.randn((27, 27), generator=g, requires_grad=True) # fills a tensor with random numbers from a normal distribution(bell curve) 

# input to the network
xenc = F.one_hot(xs, num_classes=27).float() # creates a n=27 size vector for each xs integer with a 1 on the corresponding position
#print(xenc)
#plt.imshow(xenc)

# TRAINING
for i in range(100):

    ### FORWARD PASS
    # (5, 27) @ (27, 27) -> (5, 27), for 'emma.'
    # exp() == 2**x func
    logits = xenc @ W # log-counts

    ## SOFTMAX
    counts = logits.exp() # counts, equivalent to N matrix(bigram_guide.py)
    probs = counts / counts.sum(1, keepdim=True)

    # torch.arange(n) gets n indicies 
    # ys are the indicies of the next char 
    reg_loss = 0.01*(W**2).mean() # controls the counts in (N+1), makes all weight positive
    loss = -probs[torch.arange(num), ys].log().mean() + reg_loss # average nll network assings to the next characters
    #print(loss)


    ### BACKWARD PASS
    W.grad = None # set gradients to 0
    loss.backward() # backprops calculating grads for each neuron

    W.data += -50 * W.grad # updates the tensor
    print(f'Loss: {loss.item()}')


# SAMPLING

for i in range(100):
    out = []
    ix = 0
    while True:
        # Before:
        #p = P[ix]

        # Now:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdim=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item( )

        out.append(itos[ix])
        if ix == 0: # end token
            break

    print(''.join(out))