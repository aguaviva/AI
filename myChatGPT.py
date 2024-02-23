import torch
import torch.nn as nn
from  torch.nn import functional as F
from torch.utils.data import Dataset
from omegaconf import OmegaConf
torch.manual_seed(1337)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = OmegaConf.load('./myChatGPT.yaml')

print(OmegaConf.to_yaml(config))

use_wandb = False

with open("input.txt", "r", encoding = "utf-8") as f:
    text=f.read()

vocab = list(set(text))
vocab.sort()
print("".join(vocab))
vocab_size = len(vocab)

ctoi = { vocab[i]:i for i in range(len(vocab))}
itoc = { i:vocab[i] for i in range(len(vocab))}
def encode(s): return [ ctoi[i] for i in s]
def decode(t): return "".join([ itoc[i] for i in t])

tokens = encode(text)
print(tokens[:20])
print(decode(tokens[:20]))

class CustomDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):        
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx+1:idx + self.block_size + 1]
        return x, y   

data = torch.tensor(tokens, dtype=torch.long, device = device)
n = int(0.9*len(data))
training_set = CustomDataset(data[:n], config.context_size)
validation_set  = CustomDataset(data[n:], config.context_size)
    
training_generator = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size, shuffle=True)  
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size = config.batch_size, shuffle=False)  

for i, (x,y) in enumerate(training_generator):
    print(i)
    print(x)
    print(y)
    break

dropout=0.2

def save_model(model, run_name=""):
    torch.save(model.state_dict(), f"./{model.model.name}_{run_name}.pth")    

def load_model(model, run_name=""):    
    model.load_state_dict(torch.load(f"./{model.model.name}_{run_name}.pth"))

def compute_loss(model, generator):
    model.eval()
    with torch.no_grad():
        total = 0
        for it, (x,y) in enumerate(generator):
            if it>100:
                break
            _, loss = model(x,y)
            total += loss
        model.train()
        return float((total/100).cpu())

def train(model, config = {}, notes = "", tags = []):
    if use_wandb:
        import wandb
        wandb.login()

        wandb.init(
            settings = wandb.Settings(start_method="thread"),
            # set the wandb project where this run will be logged
            project = "myChatGPT",
            notes = notes,
            tags = tags,        

            # track hyperparameters and run metadata
            config = config
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

    path = "best_model.pt"

    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except:
        pass
    
    print(compute_loss(model, training_generator), compute_loss(model, validation_generator))

    best_loss = 1e6

    for it, (x,y) in enumerate(training_generator):
        _, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        if it % config.iter_eval == 0:
            train_loss = compute_loss(model, training_generator)
            val_loss = compute_loss(model, validation_generator)            
            if use_wandb:
                wandb.log({"val_loss": val_loss, "train_loss": train_loss})
            else:
                print(it//config.iter_eval, train_loss, val_loss)

            if (val_loss<best_loss):
                best_loss = val_loss
                torch.save({
                            'config': config,
                            'iteration': it,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            }, path)                
        
    if use_wandb:
        wandb.finish()

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
    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embedding_size)

        self.head = nn.ModuleList( [Attention(config.context_size, config.embedding_size, config.embedding_size//config.num_heads) for _ in range(config.num_heads)])
        self.linear = nn.Linear(config.embedding_size, config.embedding_size)
        self.dp1 = nn.Dropout(dropout)
        
        self.ln2 = nn.LayerNorm(config.embedding_size)

        self.ff = nn.Sequential(
            nn.Linear(config.embedding_size, 4 * config.embedding_size),
            nn.ReLU(),
            nn.Linear(4 * config.embedding_size, config.embedding_size),
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
    def __init__(self):
        super().__init__()
        pos = torch.arange(0, config.context_size, dtype=torch.long)
        self.register_buffer("pos", pos)

        self.tok_embedding = nn.Embedding(vocab_size, config.embedding_size)
        self.pos_embedding = nn.Embedding(config.context_size, config.embedding_size)

        self.blocks = nn.Sequential( *[Block() for _ in range(config.num_blocks)])

        self.ln = nn.LayerNorm(config.embedding_size) # final layer norm
        self.linear = nn.Linear(config.embedding_size, vocab_size)

    def forward(self, x):
        
        te = self.tok_embedding(x)
        pe = self.pos_embedding(self.pos)
        x = te + pe

        x = self.blocks(x)
        x = self.ln(x)

        x = self.linear(x)

        return x

loss_fn = nn.CrossEntropyLoss()    

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

cg = Generator(ChatGPT()).to(device)
#cg.load_state_dict(torch.load(gen.name+".pth"))
train(cg, config, notes = "overfit", tags=[] )
print(cg.generate(1500))                