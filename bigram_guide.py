import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().split()

# (row, column)
# 26 alphabetical letters and two special chars
N = torch.zeros((27, 27), dtype=torch.int32)

# list of all chars in words
chars = sorted(list(set(''.join(words))))

# mapping of string to intiger placement
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
# reverses stoi
itos = {i:s for s,i in stoi.items()}

# Iterate over words
for w in words:
    # Makes a start and end special character for each word
    chrs = ['.'] + list(w) + ['.']

    # Iterate over a char in chrs and the next char
    for ch1, ch2 in zip(chrs, chrs[1:]):
        # Gets intiger from ch1,ch2 to ix1,ix2
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        # 2D array indexing
        N[ix1, ix2] += 1

# Gets the first row of the array (.<starting_char>)
p = N[0].float()
# Converts to probabilities
p = p / p.sum()

# Creates a generator that produces "random" numbers
g = torch.Generator().manual_seed(2147483647)

P = (N+1).float() # Copy of N, add +1 to make nothing +inf unlikely

# input=1, the 1 dimension is the rows, 0 is columns
# keepdim=True, output tensor == to size as input except in dimension(s) dim where size 1
P /= P.sum(1, keepdim=True) # BROADCASTING

for i in range(1):
    out = []
    ix = 0
    while True:
        # Set probability of the current char to the next char
        #p = N[ix].float()     
        #p = torch.ones(27) / 27 <=> equally likely distribution

        p = P[ix]

        # Generates a sample given the probabilities provided
            # replacement <=> once an element is taken, can you take it again?
            # ix <=> index
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0: # end token
            break

    print(''.join(out))


loglikelyhood = 0.0
n = 0
for w in words[:1]:
    chrs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chrs, chrs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2] # probability of ix1 followed by ix2

        # log prob to normalize (Dom:(0, 1), Rang:(-inf, 0), where x <= 1)
        logprob = torch.log(prob) 
        # log(a*b*c) == log(a) + log(b) + log(c) where a,b,c are the combined prob of a word
        loglikelyhood += logprob
        n += 1

        print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{loglikelyhood=}')
nll = -loglikelyhood # negative log likelihood to have 0 opimal model, +inf bad model
print(f'{nll/n=}') # mean of nll

#GOAL: maximize probabilities to minimize nll 