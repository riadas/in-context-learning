import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
from torch.utils.data import Dataset 
import random
import time
import datetime
import random
from transformers import GPT2LMHeadModel, GPT2Config
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import base64
from transformers import GPT2Tokenizer
#get pretrained tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', pad_token='<pad>')

import gc
gc.collect() 

# train/test data paths
train_dataset_path = sys.argv[1]
test_dataset_path = sys.argv[2]
epochs = int(sys.argv[3])

num_bits = int(train_dataset_path.split("num_bits_")[-1][0])

with open(train_dataset_path, "r") as f:
  all_sentences = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

with open(test_dataset_path, "r") as f:
  test_sentences = list(filter(lambda x: len(x) > 5, f.read().split("\n")))

all_sentences = all_sentences + test_sentences

max_len = max([len(tokenizer.encode(s)) for s in all_sentences])
print(f"max_len {max_len}")

def tokenize_seq(sent,tokenizer,max_length):
  return tokenizer(sent, truncation=True, max_length=max_length, padding="max_length")

class RationalRulesDataset(Dataset):

  def __init__(self, sentences, tokenizer, gpt2_type="gpt2", max_length=max_len):

    self.tokenizer = tokenizer 
    self.input_ids = []
    self.attn_masks = []

    for sentence in sentences:      
      encodings = tokenize_seq(sentence,tokenizer,max_length)
            
      self.input_ids.append(torch.tensor(encodings['input_ids']))
      self.attn_masks.append(torch.tensor(encodings['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]   

def format_time(elapsed):
  return str(datetime.timedelta(seconds=int(round((elapsed)))))  

#create an instance of Dataset
dataset = RationalRulesDataset(all_sentences, tokenizer, max_length=max_len)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])
print("train_size :",train_size)
print("val_size   :",val_size)

gc.collect() 

train_dataloader = DataLoader(train_set,  sampler = RandomSampler(train_set), batch_size = 32)
validation_dataloader = DataLoader(val_set, sampler = SequentialSampler(val_set), batch_size = 32 )

# Create default config
configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
# Load pretrained gpt2
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))

# Create device
device = torch.device("cuda")
model.cuda()


optimizer = torch.optim.Adam(model.parameters(),lr = 0.0005)
model = model.to(device)

def eval_on_test_data(test_sentences):
  total = 0
  weird_outputs = set()
  for i in range(len(test_sentences)):
    if i % 1000 == 0:
      print("evaluating test sentence " + str(i))
    full_sentence = test_sentences[i] # [ 1, 0, 0, 0, 1]: True, [ 0, 1, 0, 0, 1]: True, [ 0, 1, 0, 0, 0]: True, [ 1, 1, 0, 1, 0]: True, [ 1, 0, 1, 1, 1]: True, [ 1, 1, 0, 1, 1]: True
    split_full_sentence = list(map(lambda s : s if s[-1] == "e" else s + "e", full_sentence.split("e,")))
    formatted_partial_sentences = list(map(lambda i: ",".join(split_full_sentence[0: i + 1]), range(len(split_full_sentence))))
    num_correct = 0
    for sentence in formatted_partial_sentences:
      prompt = ":".join(sentence.split(":")[:-1]) + ":"
      answer = sentence.split(":")[-1].replace(" ", "")
      input_seq = prompt
      generated = torch.tensor(tokenizer.encode(input_seq)).unsqueeze(0)
      generated = generated.to(device)
      sample_outputs = model.generate(
                                  generated, 
                                  do_sample=False,   
                                  max_new_tokens = 1,
                                  num_return_sequences=1,
                                  pad_token_id=50256,
                                  )

      prediction = tokenizer.decode(sample_outputs[0], skip_special_tokens=True).split(":")[-1].replace(" ", "")
      # print("prediction: " + prediction)
      # print("answer: " + answer)
      if prediction == answer:
        num_correct += 1
      elif not prediction in ["True", "False"]:
        weird_outputs.add(prediction)
      total += num_correct/len(formatted_partial_sentences)

  return total/len(test_sentences), weird_outputs

  #call model with a batch of input
def process_one_batch(batch):
  b_input_ids = batch[0].to(device)
  b_labels = batch[0].to(device)
  b_masks = batch[1].to(device)
  outputs  = model(b_input_ids,  attention_mask = b_masks,labels=b_labels)
  return outputs

#do one epoch for training
def train_epoch():
  t0 = time.time()
  total_train_loss = 0
  model.train()
  for step, batch in enumerate(train_dataloader):
        
        model.zero_grad()        
        outputs = process_one_batch( batch)
        loss = outputs[0]  
        batch_loss = loss.item()
        total_train_loss += batch_loss

        loss.backward()
        optimizer.step()

        
  avg_train_loss = total_train_loss / len(train_dataloader)  
  print("avg_train_loss",avg_train_loss)  
  elapsed_time = format_time(time.time() - t0)
  print("elapsed time for 1 training epoch : ",elapsed_time)
  return avg_train_loss

#do one epoch for eval
def eval_epoch():
  t0 = time.time()
  total_eval_loss = 0
  nb_eval_steps = 0
  # Evaluate data for one epoch
  for batch in validation_dataloader:            
        
    with torch.no_grad():        
      outputs = process_one_batch( batch)
      loss = outputs[0]              
      batch_loss = loss.item()
      total_eval_loss += batch_loss         

  avg_val_loss = total_eval_loss / len(validation_dataloader)
  print("avg_val_loss",avg_val_loss) 
  elapsed_time = format_time(time.time() - t0)
  print("elapsed time for 1 eval epoch : ",elapsed_time)
  return avg_val_loss

train_accuracies = []
test_accuracies = []

train_losses = []
test_losses = []

# TRAINING

train_accuracy, weird_train_outputs = eval_on_test_data(all_sentences)
print("CURRENT TRAIN ACCURACY: " + str((train_accuracy, weird_train_outputs)))

test_accuracy, weird_test_outputs = eval_on_test_data(test_sentences)
print("CURRENT TEST ACCURACY: " + str((test_accuracy, weird_test_outputs)))

train_accuracies.append(train_accuracy)
test_accuracies.append(test_accuracy)
for i in range(epochs):
  print("EPOCH " + str(i))
  train_loss = train_epoch()
  test_loss = eval_epoch()
  train_accuracy, weird_train_outputs = eval_on_test_data(all_sentences)
  print("CURRENT TRAIN ACCURACY: " + str((train_accuracy, weird_train_outputs)))

  test_accuracy, weird_test_outputs = eval_on_test_data(test_sentences)
  print("CURRENT TEST ACCURACY: " + str((test_accuracy, weird_test_outputs)))

  train_accuracies.append(train_accuracy)
  test_accuracies.append(test_accuracy)

  train_losses.append(train_loss)
  test_losses.append(test_loss)

figure_path = "/".join(train_dataset_path.split("/")[:-1]) + "/"
all_acc = {"train_acc" : train_accuracies, "val_acc" : test_accuracies}
all_loss = {"train_loss" : train_losses, "val_loss" : test_losses}

print(all_acc)
plt.plot(all_acc['train_acc'], label="Train Accuracy")
plt.plot(all_acc['val_acc'], label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Train/Test Accuracy")
plt.title("Train/Test Accuracy: " + str(num_bits) + "-Bit Features")
plt.legend()
plt.grid()
plt.savefig(figure_path + "/PREFIX_ACC_plot_accuracy_num_epochs_" + str(epochs) + ".png")
plt.clf()

print(all_loss)
plt.plot(all_loss['train_loss'], label="Train Loss")
plt.plot(all_loss['val_loss'], label="Test Loss")
plt.title("Train/Test Loss: " + str(num_bits) + "-Bit Features")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.legend()
plt.grid()
plt.savefig(figure_path + "/PREFIX_ACC_plot_loss_num_epochs_" + str(epochs) + ".png")
