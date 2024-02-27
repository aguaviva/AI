import torch
from  torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def expected_x_got_y(vocab_size, model, dataset):
    res = torch.zeros([vocab_size, vocab_size], dtype=torch.long, device = device)
    model.eval()
    with torch.no_grad():
        for it, (x,y) in enumerate(dataset):
            p, _ = model(x)
            probs = F.softmax(p, dim=1)

            for i in range(probs.shape[0]):
                a = y[i,-1]
                b = torch.multinomial(probs[i],num_samples=1)[-1]

                res[a,b] +=  1
    return res

def flatten(vocab_size, dist):
    out = torch.zeros([dist.shape[0]*dist.shape[1]], dtype=torch.long)
    print(out.shape)
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            out[j*vocab_size+i] += dist[i,j].cpu()
    return out

def classes_count(vocab_size, training_set):
    cl_cnt = torch.zeros(vocab_size*vocab_size, dtype=torch.long)
    cl_id = torch.zeros(len(training_set), dtype=torch.long)
    print(len(training_set))
    for i, d in enumerate(training_set):
        x,y = d
        x=x[-1]
        y=y[-1]
        c = x*vocab_size + y
        cl_cnt[c] += 1
        cl_id[i]=c
    return cl_cnt, cl_id
