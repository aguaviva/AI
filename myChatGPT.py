import torch
import torch.nn as nn
from  torch.nn import functional as F
torch.manual_seed(1337)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import wandb
wandb.login()


with open("input.txt", "r", encoding = "utf-8") as f:
    text=f.read()
print(len(text))   

vocab = list(set(text))
vocab.sort()
print("".join(vocab))
vocab_size = len(vocab)

#encoder
ctoi = { vocab[i]:i for i in range(len(vocab))}
itoc = { i:vocab[i] for i in range(len(vocab))}
def encode(s): return [ ctoi[i] for i in s]
def decode(t): return "".join([ itoc[i] for i in t])

tokens = encode("hi ho")
s = decode(tokens)
print(tokens, s)

tokens = encode(text)
print(tokens[:20])
print(decode(tokens[:20]))

data = torch.tensor(tokens, dtype=torch.long, device = device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(data, batch_size = 4, block_size = 8 ):
    indices = torch.randint(len(data)-block_size-1, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in indices], dim=0)
    y = torch.stack([data[i+1:i+block_size+1] for i in indices], dim=0)
    return x,y   

x,y =  get_batch(train_data)    
print(x)
print(y)

def compute_loss(model, dataset):
    model.eval()
    with torch.no_grad():
        total = 0
        for i in range(100):   
            x,y =  get_batch(dataset, 64, model.get_context_size())    
            _, loss = model(x,y)
            total += loss
        model.train()
        return float((total/100).cpu())

def save_model(model, run_name=""):
    torch.save(model.state_dict(), f"./{model.model.name}_{run_name}.pth")    

def load_model(model, run_name=""):    
    model.load_state_dict(torch.load(f"./{model.model.name}_{run_name}.pth"))

def train(model, lr, batch_size, iterations, iter_eval):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(compute_loss(model, train_data), compute_loss(model, val_data))

    for it in range(iterations):
        x,y =  get_batch(train_data, batch_size, model.get_context_size())    
        _, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        if it % iter_eval == 0:
            train_loss = compute_loss(model, train_data)
            val_loss = compute_loss(model, val_data)
            print(it//iter_eval, train_loss, val_loss)
            wandb.log({"val_loss": val_loss, "train_loss": train_loss})
        


loss_fn = nn.CrossEntropyLoss()

class Attention(nn.Module):
    def __init__(self, context_size, input_size, output_size):
        super().__init__()
        # KQV size
        self.output_size = output_size
        self.key = nn.Linear(input_size, output_size, bias=False)
        self.query = nn.Linear(input_size, output_size, bias=False)
        self.value = nn.Linear(input_size, output_size, bias=False)

        sz = context_size
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask)


    def forward(self, x):
        em_key = self.key(x)
        em_query = self.query(x)
        em_value = self.value(x)

        # the attentions matrix must be the size of the context
        # as it is in reality an adjacency matrix
        att = em_query @ em_key.transpose(-2,-1)

        #print (att.shape)

        att /= self.output_size ** 0.5

        att += self.mask

        att = F.softmax(att, dim=-1)
        return att @ em_value 

class Block(nn.Module):
    def __init__(self, context_size, num_heads, embedding_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embedding_size)

        self.head = nn.ModuleList( [Attention(context_size, embedding_size, embedding_size//num_heads) for _ in range(num_heads)])
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.dp1 = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(embedding_size)

        self.ff = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout(dropout),
        )


    def forward(self, x):

        lx = self.ln1(x)
        x1 = self.linear(torch.cat([head(lx) for head in self.head], dim=-1))
        x1 = self.dp1(x1)
        x = x + x1
        
        lx = self.ln2(x)
        x2 = self.ff(lx)
        x = x + x2

        return x


class ChatGPT(nn.Module):
    def __init__(self, context_size, num_blocks, num_heads, embedding_size):
        super().__init__()
        self.name = f"gpt_{context_size}_{num_blocks}_{num_heads}_{embedding_size}"
        self.context_size = context_size
        pos = torch.arange(0, context_size, dtype=torch.long)
        self.register_buffer("pos", pos)

        self.tok_embedding = nn.Embedding(vocab_size, embedding_size)
        self.pos_embedding = nn.Embedding(context_size, embedding_size)

        self.blocks = nn.Sequential( *[Block(context_size, num_heads, embedding_size) for _ in range(num_blocks)])

        self.ln = nn.LayerNorm(embedding_size) # final layer norm
        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x):
        
        te = self.tok_embedding(x)
        pe = self.pos_embedding(self.pos)
        x = te + pe

        x = self.blocks(x)
        x = self.ln(x)

        x = self.linear(x)

        return x

class Generator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y = None):
        p = self.model(x)
        if y!=None:
            ly = F.one_hot(y, vocab_size).type(torch.float32)
            loss = loss_fn(p.permute(0,2,1), ly.permute(0,2,1))
        else:
            loss = None
        return p, loss
    
    def generate(self, count, str=" "):
        self.eval()
        with torch.no_grad():
            s = torch.zeros((1,self.model.context_size), dtype=torch.long).to(device)

            prompt = torch.tensor([encode(str)], dtype=torch.long, device = device)
            prompt_len = len(str)

            s[0, -prompt_len:] = prompt
            out = s
            for i in range(count):
                p, _ = self.forward(out[:,-self.model.context_size:])
                probs = F.softmax(p[0], dim=1)
                s = torch.multinomial(probs,1)
                out = torch.cat([out, s[-1].unsqueeze(1)], dim=1)

            return decode(out[0].tolist()[self.model.context_size-prompt_len:])
        self.train()

    def get_context_size(self):
        return self.model.context_size
    


lr = 0.001
dropout = 0.2
iterations = 7000
iter_eval=200
batch_size=64
context_size = 256
num_blocks = 6
num_heads = 6
embedding_size = 6*64
notes = "overfitting"

gen = ChatGPT(context_size, num_blocks, num_heads, embedding_size)
model = Generator(gen).to(device)

config={
"lr": lr,
"dataset": "tinyshakespeare",
"dropout" : dropout, 
"iterations": iterations,
"iter_eval": iter_eval,
"batch_size": batch_size,
"context_size": context_size,
"num_blocks": num_blocks,
"num_heads": num_heads,
"embedding_size": embedding_size
}

wandb.init(
    settings=wandb.Settings(start_method="thread"),
    # set the wandb project where this run will be logged
    project="myChatGPT",
    notes = notes,
    config = config
    # track hyperparameters and run metadata
)

train(model, lr, batch_size, iterations, iter_eval)

print(model.generate(2000))

wandb.finish()
