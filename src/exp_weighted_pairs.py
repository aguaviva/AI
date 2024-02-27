import torch
from src.myChatGPT import *
from src.metrics import *
from torch.utils.data import WeightedRandomSampler



# classes are made of two characters
# compute classes ids and counts 

c_cnt, c_id = classes_count()
print(c_cnt[0:36])        
print(c_id[0:36])   

weights = torch.zeros_like(c_id, dtype=torch.float32)
for i in range(c_id.shape[0]):
    weights[i] = 1.0/c_cnt[c_id[i]]

sampler = WeightedRandomSampler(weights, config.iterations, replacement=False)
weighted_training_generator = torch.utils.data.DataLoader(training_set, sampler=sampler, batch_size = config.batch_size)  

preds = expected_x_got_y(vocab_size, model, validation_generator)
sorted, _ = torch.sort(flatten(preds.long()), descending=True)
#plot_pairs(sorted)

#

config.name = "weighted pairs"

model = get_model()
optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

train(model, optimizer, training_generator, config)

print(model.generate(200))