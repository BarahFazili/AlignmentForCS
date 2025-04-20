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

import time
start_time = time.time()

#CONTRASTIVE ALIGNMENT FOR::BERT Sep 9 2022
#Note: This script is set by default to train entire BERT model for alignment. Use script(align_linear.py) to instead train a linear layer ontop of fixed pretrained multilingual BERT!

train_set=[] #TRAIN SET !!! (parallel sentences or MUSE word pairs in data/ folder)
file_= open('./data/?.txt', 'r')
lines = file_.readlines()
for line in lines:
  pair=(line.strip()).split("|||")
  train_set.append((pair))
print(train_set[:5])
print(len(train_set))

valid_set=[]
file_= open('./data/.txt?', 'r')

lines = file_.readlines()
for line in lines:
  pair=(line.strip()).split("|||")
  valid_set.append((pair))
print(valid_set[:5])
print(len(valid_set))

test_set=[]
file_= open('./data/.txt?', 'r')
lines = file_.readlines()
for line in lines:
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
batch_size = 8 #128 #64
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

model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.to(device)

# model_head = Model_lin()
# model_head.to(device)

loss_model = NT_Xent(batch_size,temperature=0.001,world_size=1)
loss_model.to(device)

model_fixed = BertModel.from_pretrained('bert-base-multilingual-cased')
model_fixed.to(device)
model_fixed.eval() 

# optimizer = AdamW(model.parameters(), lr=0.001, eps=1e-8, weight_decay=0.0)
optimizer = AdamW(model.parameters(), lr=0.00005, eps=1e-8, weight_decay=0.0,betas=(0.9, 0.98))

def fun(model_fixed,data):
    x,y = torch.split(data,1, dim=1)
    x = x.squeeze(dim=1);x = x.squeeze(dim=1)
    y = y.squeeze(dim=1);y = y.squeeze(dim=1)

    em_en = model_fixed(x)
    em_en = em_en[0][:,0,:]
    em_en = em_en.unsqueeze(dim=1)

    em_hi = model_fixed(y)
    em_hi = em_hi[0][:,0,:] #last hidden state for cls token
    em_hi = em_hi.unsqueeze(dim=1)

    emb_cont = torch.cat((em_en,em_hi),dim=1)
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
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='?.pt')

    #set_seed(12)
    #for epoch in range(1, n_epochs + 1):
    for epoch in tqdm (range(1, n_epochs + 1), desc="Epochs"):
        model.train()
        # model_head.train()
        for batch_idx, data in tqdm(enumerate(train_loader)):

            data = data.to(device)
            # print("Data size:",data.size()) torch.Size([BS, 2, 1, 128])
            optimizer.zero_grad()
            x,y = torch.split(data,1, dim=1)
            x = x.squeeze(dim=1);x = x.squeeze(dim=1)
            y = y.squeeze(dim=1);y = y.squeeze(dim=1)
            em_en = model(x)
            em_en = em_en[0][:,0,:]
            em_hi = model(y)
            em_hi = em_hi[0][:,0,:] #last hidden state for cls token
            #pass through the linear layer
            # output = model_head(em_en, em_hi)
            output = torch.cat((em_en.unsqueeze(1),em_hi.unsqueeze(1)),dim=1)
            criterion = loss_model
            # loss = criterion(em_en, em_hi)
            loss = criterion(output[:,0,:], output[:,1,:])
            with torch.no_grad():
                original = fun(model_fixed,data)
            # print("Original size:",original.size()) #torch.Size([BS, 2, 768])
            # print("Output size:",output.size()) #torch.Size([BS, 2, 768])
            loss += scaling*reg_loss(output,original)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
    
        model.eval() # prep model for evaluation
        # model_head.eval()
        for batch_idx, data in enumerate(valid_loader):
            data = data.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            x,y = torch.split(data,1, dim=1)
            x = x.squeeze(dim=1);x = x.squeeze(dim=1)
            y = y.squeeze(dim=1);y = y.squeeze(dim=1)
            em_en = model(x)
            em_en = em_en[0][:,0,:]
            em_hi = model(y)
            em_hi = em_hi[0][:,0,:] #last hidden state for cls token
            #pass through the linear layer
            # output = model_head(em_en, em_hi)
            output = torch.cat((em_en.unsqueeze(1),em_hi.unsqueeze(1)),dim=1)
            criterion = loss_model
            loss = criterion(output[:,0,:], output[:,1,:])
            with torch.no_grad():
                original = fun(model_fixed,data)
            # print("Original size:",original.size()) torch.Size([BS, 2, 768])
            # print("Output size:",output.size()) torch.Size([BS, 2, 768])
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
        
        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if epoch == 1: # or epoch == 5:
            save_path='./?'+str(epoch)+'.pt'
            torch.save(model.state_dict(), save_path)

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
        em_en = model(x)
        em_en = em_en[0][:,0,:]
        #with torch.no_grad():
        em_hi = model(y)
        em_hi = em_hi[0][:,0,:] #last hidden state for cls token
        #passing through the linear layer
        # output = model_head(em_en, em_hi)
        output = torch.cat((em_en.unsqueeze(1),em_hi.unsqueeze(1)),dim=1)
        
        for (x,y) in output:
            count += 1
            sim = cos(x,y).item()
            sim_sum += sim
            cos_sim_list.append(sim)
        
    print("*******************Average cosine similarity on test set: ",sim_sum/count)
    file_ = open("./?.txt", "w+")  # write mode 
    
    for x in cos_sim_list:
        file_.write(str(x)+ '\n')
    file_.close() 


n_epochs = 10

# early stopping patience; how long to wait after last time validation loss improved.
patience = 10
scaling= 0.1
model,train_loss, valid_loss = train(batch_size,patience,n_epochs,scaling)





# SAVE MODEL AFTER TRAINING
# save_path='?.pt'
# torch.save(model.state_dict(), save_path)
# save_path='?.pt'
# torch.save(model_head.state_dict(), save_path)

#LOAD MODEL FOR EVALUATION

save_path='./fullmBERT/contrastive/9/bert.pt'
model = BertModel.from_pretrained('bert-base-multilingual-cased')
model.load_state_dict(torch.load(save_path, map_location='cpu'))
model.to(device)
# save_path='?.pt'
# model_head = Model_lin()
# model_head.load_state_dict(torch.load(save_path, map_location='cpu'))
# model_head.to(device)

model.eval()
# model_head.eval()
with torch.no_grad():
    test(model)

print("--- %s seconds ---" % (time.time() - start_time))
