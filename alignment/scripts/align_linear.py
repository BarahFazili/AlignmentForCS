#Instead of cls embeddings we align the pooled output.
#En-Es
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping
from transformers import BertTokenizer, BertModel, AdamW
from tqdm import tqdm
from functools import partial

#CONTRASTIVE ALIGNMENT FOR::BERT+LINEAR LAYER (FIXED BERT)  Sep 9 2022
#Note: This script trains a linear layer on top of fixed mBERT for multilingual alignment. Use script(align_bert.py) to instead train multilingual BERT!


import time
start_time = time.time()


#TRAIN SET !!! (parallel sentences or MUSE word pairs in data/ folder)

train_set=[] 
file_= open('?.txt', 'r')
lines = file_.readlines()
for line in lines:
#   pair=(line.strip()).split("|||")
  pair=(line.strip()).split("|||")
  train_set.append((pair))
print(train_set[:5])
print(len(train_set))

valid_set=[]
file_= open('./?.txt', 'r')
lines = file_.readlines()
for line in lines:
#   pair=(line.strip()).split("|||")
  pair=(line.strip()).split("|||")
  valid_set.append((pair))
print(valid_set[:5])
print(len(valid_set))

test_set=[]
file_= open('./?', 'r') #same as valid
lines = file_.readlines()
for line in lines:
#   pair=(line.strip()).split("|||")
  pair=(line.strip()).split("|||")
  test_set.append((pair))
print(test_set[:5])
print(len(test_set))



def collate_fn(batch_,tokenizer):
  batch=[]
  for en,hi in batch_:
    encoded_en = tokenizer.encode_plus(
    text=en,  # the sentence to be encoded
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length = 128 , #16,  # maximum length of a sentence
    pad_to_max_length=True,  # Add [PAD]s
    truncation=True,
    return_attention_mask = False,  # Generate the attention mask
    return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )
    input_ids_en = encoded_en['input_ids']#.to(device)
    #attn_mask_en = encoded_en['attention_mask']
    #e_bert=torch.cat((input_ids_en,attn_mask_en),dim=0).unsqueeze(0)
    #print(e_bert.size())

    encoded_hi = tokenizer.encode_plus(
    text=hi,  # the sentence to be encoded
    add_special_tokens=True,  # Add [CLS] and [SEP]
    max_length = 128,#64,  # maximum length of a sentence
    pad_to_max_length=True,  # Add [PAD]s
    truncation=True,
    return_attention_mask = False,  # Generate the attention mask
    return_tensors = 'pt',  # ask the function to return PyTorch tensors
    )
    input_ids_hi = encoded_hi['input_ids']#.to(device)
    #attn_mask_hi = encoded_hi['attention_mask']
    #h_bert=torch.cat((input_ids_hi,attn_mask_hi),dim=0).unsqueeze(0)
    #print(h_bert.size())

    #pair=torch.cat((h_bert,e_bert),dim=0)
    pair=torch.cat((input_ids_en.unsqueeze(0),input_ids_hi.unsqueeze(0)),dim=0)
    pair=pair.to(device)
    # print(pair.size()) #2,1,16
    batch.append(pair)
  tensors=torch.stack(batch)
  #tensors=tensors.to(device)
  #print("********Batch size from collate_fn:********")
  #print(tensors.size()) #32,2,1,16
  return tensors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:",device)

if device == "cuda:0":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
batch_size = 512 #128 #64
 #512

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tokenizer),
    num_workers=num_workers,
    pin_memory=pin_memory,
)

valid_loader = torch.utils.data.DataLoader(
    valid_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=partial(collate_fn, tokenizer=tokenizer),
    #sampler=valid_sampler,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    collate_fn=partial(collate_fn, tokenizer=tokenizer),
    num_workers=num_workers,
    pin_memory=pin_memory,
)

# test_loader = torch.utils.data.DataLoader(
#     PT, #test_set,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     collate_fn=partial(collate_fn, tokenizer=tokenizer),
#     num_workers=num_workers,
#     pin_memory=pin_memory,
# )

class Model_lin(nn.Module):
  def __init__(self):
    super().__init__() 
    self.cont_head = nn.Linear(768,768)    
  def forward(self, em_en, em_hi): 
    em_en = self.cont_head(em_en)
    em_en = em_en.unsqueeze(dim=1)
    em_hi = self.cont_head(em_hi)
    em_hi = em_hi.unsqueeze(dim=1)
    emb_cont = torch.cat((em_en,em_hi),dim=1)
    return emb_cont


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=-1) 

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        #For later excluding negative pairs from same language
        for i in range(batch_size):
            for j in range(batch_size):
                mask[i, j] = 0
                mask[i+batch_size, j+batch_size] = 0
        return mask

    def forward(self, z_i, z_j):
        # print("check 1:",z_i.size())
        batch_size=z_i.shape[0]
        N = 2 * batch_size * self.world_size
        z = torch.cat((z_i, z_j), dim=0)
        # print("check 2:",z.size())
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        # print("check 3:",sim.size())
        sim_i_j = torch.diag(sim, batch_size * self.world_size)
        # print("check 4:",sim_i_j.size())
        sim_j_i = torch.diag(sim, -batch_size * self.world_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # print("check 5:",positive_samples.size())
        mask = self.mask_correlated_samples(batch_size, world_size=1)
        # print("check 6:",mask.size())
        negative_samples = sim[mask].reshape(N, -1)
        # print("check 7:",negative_samples.size())
        labels = torch.zeros(N).to(positive_samples.device).long()
        # print("check 8:",labels.size())
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        # print("check 10:",logits.size())
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

def reg_loss(input_, output):
  cos = nn.CosineSimilarity(dim = -1)
  d = 1 - cos(input_,output)
  loss = torch.mean(d)
  return loss

model = Model_lin()
model.to(device)

loss_model = NT_Xent(batch_size,temperature=0.01,world_size=1)
loss_model.to(device)

model_fixed = BertModel.from_pretrained('bert-base-multilingual-cased')
model_fixed.to(device)
model_fixed.eval()  #added this!

optimizer = AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.0,betas=(0.9, 0.98))
# optimizer = AdamW(model.parameters(), lr=0.00005, eps=1e-8, weight_decay=0.0,betas=(0.9, 0.98))

#optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
## scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.1)  # reduce the learning after 20 epochs by a factor of 10
#no_decay = ["bias", "LayerNorm.weight"]
# optimizer_grouped_parameters = [
#         {"params":[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.0001},
#         {"params":[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0001},
#         # {"params":[p for n,p in cont_head.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":0.0},
#         # {"params":[p for n,p in cont_head.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0},
#     ]
#optimizer = AdamW(optimizer_grouped_parameters, lr=0.001, eps=1e-8)

def project(model_fixed,data):
    x,y = torch.split(data,1, dim=1)
    x = x.squeeze(dim=1);x = x.squeeze(dim=1)
    y = y.squeeze(dim=1);y = y.squeeze(dim=1)

    em_en = model_fixed(x)
    em_en = em_en[1] #BS*768
    em_en = em_en.unsqueeze(dim=1)#.to(device);

    em_hi = model_fixed(y)
    em_hi = em_hi[1]
    em_hi = em_hi.unsqueeze(dim=1)#.to(device);

    emb_cont = torch.cat((em_en,em_hi),dim=1)#.to(device);
    return emb_cont


def train(batch_size,patience,n_epochs,scaling):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='./?.pt')

    #set_seed(12)
    #for epoch in range(1, n_epochs + 1):
    for epoch in tqdm (range(1, n_epochs + 1), desc="Epochs"):
        model.train()
        for batch_idx, data in tqdm(enumerate(train_loader)):

            data = data.to(device)
            # print("Data size:",data.size())
            optimizer.zero_grad()
            x,y = torch.split(data,1, dim=1)
            x = x.squeeze(dim=1);x = x.squeeze(dim=1)
            y = y.squeeze(dim=1);y = y.squeeze(dim=1)
            with torch.no_grad():
                em_en = model_fixed(x)
                em_en = em_en[1]  #we're aligning the pooled output !
                # print("check 1",em_en.size())
            with torch.no_grad():
                em_hi = model_fixed(y)
                em_hi = em_hi[1]

            output = model(em_en, em_hi)
            # print("Model output size:",output.size())
            criterion = loss_model
            loss = criterion(output[:,0,:], output[:,1,:])
            #print("loss XENT:",loss)
            with torch.no_grad():
                original = project(model_fixed,data)
            # print("Original size:",original.size())
            loss += scaling*reg_loss(output,original)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
    
        model.eval() # prep model for evaluation
        for batch_idx, data in enumerate(valid_loader):
            data = data.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            x,y =torch.split(data,1, dim=1)
            x = x.squeeze(dim=1);x = x.squeeze(dim=1)
            y = y.squeeze(dim=1);y = y.squeeze(dim=1)
            with torch.no_grad():
                em_en = model_fixed(x)
            em_en = em_en[1]
            with torch.no_grad():
                em_hi = model_fixed(y)
            em_hi = em_hi[1]
            
            output = model(em_en, em_hi)
            criterion = loss_model
            loss = criterion(output[:,0,:],output[:,1,:])
            with torch.no_grad():
                original = project(model_fixed,data)
            loss += scaling*reg_loss(output,original)
            valid_losses.append(loss.item())
    
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('./?.pt'))
    return  model, avg_train_losses, avg_valid_losses



def test(model):
    cos_sim_list = []
    sim_sum = 0
    cos = nn.CosineSimilarity(dim=-1)
    count = 0
    #for data in test_loader: 
    for batch_idx, data in tqdm(enumerate(test_loader)):
        data = data.to(device)
        x,y = torch.split(data,1, dim=1)
        x = x.squeeze(dim=1);x = x.squeeze(dim=1)
        y = y.squeeze(dim=1);y = y.squeeze(dim=1)
        #with torch.no_grad():
        em_en = model_fixed(x)
        em_en = em_en[1]
        #with torch.no_grad():
        em_hi = model_fixed(y)
        em_hi = em_hi[1]
        #passing through the linear layer
        output = model(em_en, em_hi)
        for (x,y) in output:
            count += 1
            sim = cos(x,y).item()
            sim_sum += sim
            cos_sim_list.append(sim)
        
    print("*******************Average cosine similarity on test set: ",sim_sum/count)
    file_ = open("./?.txt", "w+")  # write modePOS 
    for x in cos_sim_list:
        file_.write(str(x)+ '\n')
    file_.close() 


n_epochs = 100

# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
scaling = 100
model, train_loss, valid_loss = train(batch_size,patience,n_epochs,scaling)



# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss))+1 
plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 10) # consistent scale
plt.xlim(0, len(train_loss)+1) # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('./?.png', bbox_inches='tight')

# # SAVE MODEL AFTER TRAINING
# save_path='./?.pt'
# torch.save(model.state_dict(), save_path)


# #LOAD MODEL FOR EVALUATION

save_path = './?.pt'
model = Model_lin()
model.load_state_dict(torch.load(save_path, map_location='cpu'))
model.to(device)



model.eval()
with torch.no_grad():
    test(model)

print("--- %s seconds ---" % (time.time() - start_time))
