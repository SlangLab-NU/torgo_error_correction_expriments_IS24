#!/usr/bin/env python
# coding: utf-8

# In[1]:

import re
import torch
import zipfile
import pandas as pd

from huggingface_hub import Repository
from datasets import load_dataset, DatasetDict, Dataset, Audio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2ProcessorWithLM
from pyctcdecode import build_ctcdecoder
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from tqdm import tqdm
from evaluate import load
from datetime import datetime
import argparse


# In[2]:
parser = argparse.ArgumentParser(description="Script to process data with Wav2Vec2 model")
parser.add_argument("--test_speaker", type=str, default="M02", help="Test speaker ID")
parser.add_argument("--split", type=str, default="test", help="Data split (e.g., 'test')")
parser.add_argument("--level", type=str, default="word", help="Processing level (e.g., 'word')")
parser.add_argument("--pattern", type=str, default="no_keep", help="Pattern (e.g., 'no_keep')")
parser.add_argument("--identifier", type=str, default="euroLM", help="Identifier")
parser.add_argument("--arpa_dir", type=str, default="europarl_lm_arpa_files", help="Directory for ARPA files")
parser.add_argument("--ngram_order", type=int, default=3, help="N-gram order")
parser.add_argument("--target_lang", type=str, default="en", help="Target language")
parser.add_argument("--text_count_threshold", type=int, default=40, help="Text count threshold")
parser.add_argument("--model_user", type=str, default="macarious", help="Model user")
parser.add_argument("--kenlm_model", type=str, default="3gram.bin", help="Path to KenLM model file")
args = parser.parse_args()

test_speaker = args.test_speaker
split = args.split
level = args.level
pattern = args.pattern
identifier = args.identifier
arpa_dir = args.arpa_dir
ngram_order = args.ngram_order
target_lang = args.target_lang
text_count_threshold = args.text_count_threshold
model_user = args.model_user
kenlm_model = args.kenlm_model 

ngram_order = 3
target_lang="en"
text_count_threshold = 40
model_user = "macarious"


if pattern == "no_keep":
  model_repo = f"torgo_xlsr_finetune_{test_speaker}_old"
elif pattern == "keep_all":
  model_repo = f"torgo_xlsr_finetune_{test_speaker}_keep_all"
elif pattern == "new":
  model_repo = f"torgo_xlsr_finetune_{test_speaker}"
    
    
model_repo_path = f"{model_user}/{model_repo}"

lm_local_path = arpa_dir


print(f"results_{identifier}_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}.txt")
print(f"model: {model_repo_path} ")
print(f"kenlm: {kenlm_model} ")

# In[3]:


# # Download the trained model
processor = Wav2Vec2Processor.from_pretrained(model_repo_path)
model = Wav2Vec2ForCTC.from_pretrained(model_repo_path)


# In[4]:


# Read the dataset
data_df = pd.read_csv('torgo_new.csv')
dataset_csv = load_dataset('csv', data_files='torgo_new.csv')

speakers = data_df['speaker_id'].unique()

print(f'Speakers: {", ".join(speakers)}')


# In[5]:


# Split data into train, valid, test sets
valid_speaker = 'F03' if test_speaker != 'F03' else 'F04'
train_speaker = [s for s in speakers if s not in [test_speaker, valid_speaker]]

torgo_dataset = DatasetDict()
torgo_dataset['train'] = dataset_csv['train'].filter(lambda x: x in train_speaker, input_columns=['speaker_id'])
torgo_dataset['validation'] = dataset_csv['train'].filter(lambda x: x == valid_speaker, input_columns=['speaker_id'])
torgo_dataset['test'] = dataset_csv['train'].filter(lambda x: x == test_speaker, input_columns=['speaker_id'])

torgo_dataset


# In[6]:


# Count the number of times the text has been spoken in each of the 'train',
# 'validation', and 'test' sets. Remove text according to the
# text_count_threshold from a previous cell.

if pattern == "no_keep":
    unique_texts = set(torgo_dataset['train'].unique(column='text')) | set(torgo_dataset['validation'].unique(column='text')) | set(torgo_dataset['test'].unique(column='text'))
    unique_texts_count = {}
    
    for text in unique_texts:
      unique_texts_count[text] = {'train_validation': 0, 'test': 0}
    
    for text in torgo_dataset['train']['text']:
      unique_texts_count[text]['train_validation'] += 1
    
    for text in torgo_dataset['validation']['text']:
      unique_texts_count[text]['train_validation'] += 1
    
    for text in torgo_dataset['test']['text']:
      unique_texts_count[text]['test'] += 1
    
    texts_to_keep_in_train_validation = []
    texts_to_keep_in_test = []
    for text in unique_texts_count:
      if unique_texts_count[text]['train_validation'] < text_count_threshold and unique_texts_count[text]['test'] > 0:
        texts_to_keep_in_test.append(text)
      else:
        texts_to_keep_in_train_validation.append(text)
    
    original_data_count = {'train': len(torgo_dataset['train']), 'validation': len(torgo_dataset['validation']), 'test': len(torgo_dataset['test'])}
    
    # Update the three dataset splits
    torgo_dataset['train'] = torgo_dataset['train'].filter(lambda x: x['text'] in texts_to_keep_in_train_validation)
    torgo_dataset['validation'] = torgo_dataset['validation'].filter(lambda x: x['text'] in texts_to_keep_in_train_validation)
    torgo_dataset['test'] = torgo_dataset['test'].filter(lambda x: x['text'] in texts_to_keep_in_test)
    

elif pattern == "keep_all":
    pass
elif pattern == "new":
        # Update the three dataset splits (if ['test_data'] == 1, keep in test, if ['test_data'] == 0, keep in train and validation)
    torgo_dataset['train'] = torgo_dataset['train'].filter(
        lambda x: x['test_data'] == 0)
    torgo_dataset['validation'] = torgo_dataset['validation'].filter(
        lambda x: x['test_data'] == 0)
    torgo_dataset['test'] = torgo_dataset['test'].filter(
        lambda x: x['test_data'] == 1)

        # Drop the 'test_data' column
    torgo_dataset['train'] = torgo_dataset['train'].remove_columns([
                                                                   'test_data'])
    torgo_dataset['validation'] = torgo_dataset['validation'].remove_columns([
                                                                             'test_data'])
    torgo_dataset['test'] = torgo_dataset['test'].remove_columns([
                                                                 'test_data'])


# Functions to process data:

# Remove special characters and convert all text into lowercase
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"0-9]'
sampling_rate=16000

def remove_special_characters(batch):
    batch['text'] = re.sub(chars_to_ignore_regex, ' ', batch['text']).lower()
    return batch

def prepare_torgo_dataset(batch):
    # Load audio data into batch
    audio = batch['audio']

    # Extract values
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    # Encode to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# In[8]:


if split == "test":
    torgo_test_set = torgo_dataset['test']
else:
    torgo_test_set = torgo_dataset['validation']

# Remove special characters
torgo_test_set = torgo_test_set.map(remove_special_characters)


if level == 'sentence':
    torgo_test_set = torgo_test_set.filter(lambda example: len(example["text"].split()) > 1)
    print(f"actual sample size: {torgo_test_set}")
elif level == "word":
    torgo_test_set = torgo_test_set.filter(lambda example: len(example["text"].split()) == 1)
    print(f"actual sample size: {torgo_test_set}")

# Filter audio within a certain length
torgo_test_set = torgo_test_set.cast_column("audio", Audio(sampling_rate=sampling_rate))
torgo_test_set = torgo_test_set.map(
  prepare_torgo_dataset,
  remove_columns=['session', 'audio', 'speaker_id'],
  num_proc=4
)

min_input_length_in_sec = 1.0
max_input_length_in_sec = 10.0
torgo_test_set = torgo_test_set.filter(lambda x: x < max_input_length_in_sec * sampling_rate, input_columns=["input_length"])
torgo_test_set = torgo_test_set.filter(lambda x: x > min_input_length_in_sec * sampling_rate, input_columns=["input_length"])

print()
print(len(torgo_test_set))


# In[12]:


def evaluateModel(processor, model, dataset, lm_model_path=None):

  predictions = []
  references = []

  if not lm_model_path:
    for i in tqdm(range(dataset.num_rows)):
      inputs = processor(dataset[i]["input_values"], sampling_rate=sampling_rate, return_tensors="pt")
      with torch.no_grad():
        logits = model(**inputs).logits
      predicted_ids = torch.argmax(logits, dim=-1)
      transcription = processor.batch_decode(predicted_ids)

      predictions.append(transcription[0].lower())
      references.append(dataset[i]["text"])

  else:
    vocab_dict = processor.tokenizer.get_vocab()
    sorted_vocab_dict = {k: v for k, v in sorted(
        vocab_dict.items(), key=lambda item: item[1])}

    unigrams = set()

    with open(f"{lm_local_path}/unigrams.txt", "r") as f:
      for line in f:
        line = line.strip()
        unigrams.add(line)

    # Implement language model in the decoder
    decoder = build_ctcdecoder(
        labels=list(sorted_vocab_dict.keys()),
        kenlm_model_path=lm_model_path if ngram_order > 1 else None,
        unigrams=unigrams
    )

    # Build new processor with new decoder
    processor = Wav2Vec2ProcessorWithLM(
        feature_extractor=processor.feature_extractor,
        tokenizer=processor.tokenizer,
        decoder=decoder
    )

    # Transcripe the audio
    for i in tqdm(range(dataset.num_rows)):
      inputs = processor(dataset[i]["input_values"], sampling_rate=sampling_rate, return_tensors="pt")
      with torch.no_grad():
        logits = model(**inputs).logits

      transcription = processor.batch_decode(logits.numpy()).text

      predictions.append(transcription[0].lower())
      references.append(dataset[i]["text"])

  # Calculate the wer score
  wer = load("wer")
  wer_score = wer.compute(predictions=predictions, references=references)

  return wer_score, predictions, references


# In[13]:


wer_score_no_lm, predictions_no_lm, references_no_lm = evaluateModel(processor, model, torgo_test_set)

print(f"WER (no LM): {wer_score_no_lm}")


# In[14]:


wer_score_lm, predictions_lm, references_lm = evaluateModel(processor, model, torgo_test_set, f"{lm_local_path}/{kenlm_model}")

print(f"WER ({ngram_order}-gram): {wer_score_lm}")


# In[ ]:


# unigrams = set()

# with open(f"{lm_local_path}/unigrams.txt", "r") as f:
#   for line in f:
#     line = line.strip()
#     unigrams.add(line)

# print(len(set("".join(unigrams))))
# print(set("".join(unigrams)))
# print(unigrams)


# In[ ]:


import csv


filename=f"results_{identifier}_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}"
# Save results to a csv file
with open(f"{filename}.txt", "w") as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(["Prediction (no LM)", f"Prediction ({ngram_order}-gram)", "Reference"])
  for i in range(len(predictions_no_lm)):
    csv_writer.writerow([predictions_no_lm[i], predictions_lm[i], references_lm[i]])


# In[ ]:


# Save wer to a csv file

with open(f"wer_{ngram_order}gram_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}.txt", "w") as csv_file:
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(["Language Model", "WER"])
  csv_writer.writerow(["None", wer_score_no_lm])
  csv_writer.writerow([f"{ngram_order}-gram", wer_score_lm])

# # Display as dataframe
# results_wer_df = pd.read_csv(f"wer_{ngram_order}gram_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}.txt")
# results_wer_df.head(20)


# In[ ]:


# Create a string of current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Zip the results into a single file for download
output_zip_path = f"results_with_LM_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}_{current_date}.zip"
with zipfile.ZipFile(output_zip_path, "w") as zip_file:
  zip_file.write(f"results_{ngram_order}gram_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}.txt")
  zip_file.write(f"wer_{ngram_order}gram_{test_speaker}_{test_speaker}_{level}_{pattern}_{split}.txt")

files.download(output_zip_path)


# In[ ]:




