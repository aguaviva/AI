from tqdm import tqdm
import torch
import torch.nn as nn
from  torch.nn import functional as F
from torch.utils.data import Dataset, random_split
from omegaconf import OmegaConf
from model import Generator

torch.manual_seed(1337)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

config = OmegaConf.load('./config/myChatGPT.yaml')


class Tokenizer:
    def __init__(self, text):
        self.vocab = list(set(text))
        self.vocab.sort()

        self.ctoi = { self.vocab[i]:i for i in range(len(self.vocab)) }
        self.itoc = { i:self.vocab[i] for i in range(len(self.vocab)) }

    def get_vocab_size(self):
        return len(self.vocab)

    def encode(self, s): 
        return [ self.ctoi[i] for i in s ]

    def decode(self, t): 
        return "".join([ self.itoc[i] for i in t ])


class CustomDataset(Dataset):
    def __init__(self, tokens, context_size):
        self.data = torch.tensor(tokens, dtype=torch.long, device = device)
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __getitem__(self, idx):        
        if idx >= len(self): raise IndexError   
        x = self.data[idx:idx + self.context_size]
        y = self.data[idx+1:idx + self.context_size+1]
        return x, y   


def compute_loss(model, generator):
    model.eval()
    with torch.no_grad():
        total = 0
        with tqdm(total=len(generator)) as pbar:
            for x,y in generator:
                _, loss = model(x,y)
                total += loss
                pbar.update(1)
        model.train()
        return float((total/len(generator.dataset)).cpu())

def train(model, optimizer, dataset, config = {}):
    if config.use_wandb:
        import wandb
        wandb.login()

        wandb.init(
            settings = wandb.Settings(start_method="thread"),
            # set the wandb project where this run will be logged
            project = config.project,
            notes = config.notes,
            tags = config.tags,        

            # track hyperparameters and run metadata
            #config = config
        )

    generator = torch.Generator().manual_seed(42)
    training_set, validation_set = random_split(dataset, [0.9, 0.1], generator=generator)

    training_generator = torch.utils.data.DataLoader(training_set, batch_size = config.batch_size, shuffle=True)  
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size = config.batch_size, shuffle=False)  

    if config.resume_training:
        checkpoint = torch.load(config.model_path)
        print(f" train_loss: {checkpoint['train_loss']}")
        print(f" val_loss: {checkpoint['val_loss']}")

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])       

    best_loss = 1e6

    for it, (x,y) in enumerate(training_generator):
        if it>config.iterations:
            break

        _, loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

        if (it == 0) or (it % config.iter_eval) == 0:
            train_loss = compute_loss(model, training_generator)
            val_loss = compute_loss(model, validation_generator)            
            if config.use_wandb:
                wandb.log({"loss/val": val_loss, "loss/train": train_loss})
            
            print(f"step {it//config.iter_eval}: train loss: {train_loss},  val_loss: {val_loss}")

            if (val_loss<best_loss):
                best_loss = val_loss
                torch.save({
                            'config': config,
                            'iteration': it,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'train_loss': train_loss,
                            'rng_state': torch.get_rng_state()
                            }, config.model_path)                

    if config.use_wandb:
        wandb.finish()

if __name__ == '__main__':

    with open("data/input.txt", "r", encoding = "utf-8") as f:
        text=f.read()

    tok = Tokenizer(text)

    dataset = CustomDataset(tok.encode(text), config.context_size)

    model = Generator(tok.get_vocab_size(), config).to(device) 

    print("trainable parameter count: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

    train(model, optimizer, dataset, config)
    """
    from torch.profiler import profile, record_function, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            #train(model, optimizer, dataset, config)
            print(tok.decode(model.generate(200, tok.encode(" "))))
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace("trace.json")
    """
    print(tok.decode(model.generate(200, tok.encode(" "))))
   
    
