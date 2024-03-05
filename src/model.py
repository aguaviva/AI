import torch
import torch.nn as nn
from  torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    def __init__(self, embedding_size, context_size, num_heads, dropout):
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

    def __init__(self, vocab_size, config):

        super().__init__()
        pos = torch.arange(0, config.context_size, dtype=torch.long)
        self.register_buffer("pos", pos)

        self.tok_embedding = nn.Embedding(vocab_size, config.embedding_size)
        self.pos_embedding = nn.Embedding(config.context_size, config.embedding_size)

        self.blocks = nn.Sequential( *[Block(config.embedding_size, config.context_size, config.num_heads, config.dropout) for _ in range(config.num_blocks)])

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

class Generator(nn.Module):
    
    def __init__(self, vocab_size, config):
        super().__init__()
        self.model = ChatGPT(vocab_size, config)
        self.vocab_size = vocab_size
        self.context_size = config.context_size

    def forward(self, x, y = None):
        logits = self.model(x)
        if y!=None:
            ly = F.one_hot(y, self.vocab_size).type(torch.float32)
            loss = F.cross_entropy(logits.permute(0,2,1), ly.permute(0,2,1))
        else:
            loss = None
        return logits, loss

    def generate(self, count, tokens):
        self.eval()
        with torch.no_grad():
            s = torch.zeros((1, self.context_size), dtype=torch.long)

            prompt = torch.tensor([tokens], dtype=torch.long, device = device)
            prompt_len = len(tokens)

            s[0, -prompt_len:] = prompt
            out = s
            for i in range(count):
                p, _ = self.forward(out[:,-self.context_size:])
                probs = F.softmax(p, dim=-1)
                s = torch.multinomial(probs[0],1)
                out = torch.cat([out, s[-1].unsqueeze(1)], dim=1)

            self.train()
            return out[0].tolist()[self.context_size - prompt_len:]
