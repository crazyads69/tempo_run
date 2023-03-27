from sklearn.metrics import accuracy_score
import torch.nn as nn
from preprocess import *
import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import re
import string
import demoji
"""
    Prepare train, validation, test dataset
"""
demoji.download_codes()

translator = str.maketrans("", "", string.punctuation)
clean_csv()
train_data = prepare_train_set()
train_label = prepare_train_label()
val_data = prepare_val_set()
val_label = prepare_val_label()
test_data = prepare_test_set()
test_label = prepare_test_label()
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

stop_words = ['doubledot', 'sub', 'dot', 'add', 'fraction', 'multiply',
              'và', 'là', 'của', 'cho', 'được', 'trong', 'từ', 'nhưng', 'với', 'tại']


"""
    Define hyperparameter
"""

vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 64
num_layers = 6
dropout_prob = 0.2

def remove_stopwords(sentence, stop_words):
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    text = ' '.join(filtered_words)
    return text


def clean_text(text):
    text = re.sub(r'\bcolon\w+\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\bwzjwz\w+\b', '', text)
    text = remove_stopwords(text, stop_words)
    text = demoji.replace(text, '')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(translator)
    return text


"""
    Create Dataset class for train, validation, test
"""


class SentenceDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.float),
        }


train_dataset = SentenceDataset(train_data, train_label, tokenizer)
val_dataset = SentenceDataset(val_data, val_label, tokenizer)
test_dataset = SentenceDataset(test_data, test_label, tokenizer)

"""
    Create Dataloader to load dataset for train, validation, test
"""

train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(
    val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False)

"""
    Define Bi-LSTM with attention layer and dropout layer
"""


class BiLSTMModel(pl.LightningModule):
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout_prob):
        super().__init__()
        """
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        """
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        for param in self.phobert.parameters():
            param.requires_grad = False
        self.bilstm = nn.LSTM(self.phobert.config.hidden_size, hidden_dim,
                              num_layers=num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        """
        embedded = self.embedding(input_ids)
        
        embedded = self.phobert(input_ids, attention_mask)[0]
        embedded = self.dropout(embedded)
        """
        with torch.no_grad():
            embedded = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            embedded_output = embedded.last_hidden_state
        # detach sequence_output to prevent gradients from being computed on PhoBERT
        embedded_output = embedded_output.detach()
        embedded_output = self.dropout(embedded_output)
        outputs, _ = self.bilstm(embedded_output)
        outputs = self.dropout(outputs)
        attention_logits = self.attention(outputs)
        attention_logits = attention_logits.masked_fill(
            attention_mask.unsqueeze(-1) == 0, -1e9)
        attention_weights = F.softmax(attention_logits, dim=1)
        weighted_outputs = torch.sum(outputs * attention_weights, dim=1)
        dense_outputs = F.relu(self.fc1(weighted_outputs))
        logits = self.fc2(dense_outputs)
        return logits.squeeze()

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, prog_bar=True,
                 on_step=True, on_epoch=True)
        self.training_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_train_epoch_end(self):
        train_loss = torch.stack([x['loss']
                                 for x in self.training_step_outputs]).mean()
        self.log('train_loss_epoch', train_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.training_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            if 'logits' in output:
                y_pred.append(
                    float(torch.sigmoid(output['logits']).item() >= 0.5))

        acc = accuracy_score(y_true, y_pred)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(
            logits, labels)
        self.log('val_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.validation_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_validation_epoch_end(self):
        val_loss = torch.stack([x['loss']
                                for x in self.validation_step_outputs]).mean()
        self.log('val_loss_epoch', val_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.validation_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            if 'logits' in output:
                y_pred.append(
                    float(torch.sigmoid(output['logits']).item() >= 0.5))
        acc = accuracy_score(y_true, y_pred)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(
            logits, labels)
        self.log('test_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.test_step_outputs.append(
            {'loss': loss, 'label': labels, 'logits': logits})
        return loss

    def on_test_epoch_end(self):
        test_loss = torch.stack([x['loss']
                                for x in self.test_step_outputs]).mean()
        self.log('test_loss_epoch', test_loss, prog_bar=True, on_epoch=True)
        y_true = []
        y_pred = []
        for output in self.test_step_outputs:
            if 'label' in output:
                y_true.append(output['label'].item())
            if 'logits' in output:
                y_pred.append(
                    float(torch.sigmoid(output['logits']).item() >= 0.5))
        acc = accuracy_score(y_true, y_pred)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def predict(self, sentence):
        # Tokenize the input sentence
        sentence = clean_text(sentence)
        print(sentence)
        encoded_dict = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        # Pass the input sequence through the model to get the predicted logits
        logits = self(input_ids, attention_mask)
        # Apply a sigmoid function to the logits to get the predicted probabilities
        probs = torch.sigmoid(logits)
        # Round the probabilities to get the predicted labels
        avg = probs.mean().item()
        print("Confidence:",avg)
        # Return the predicted label (0 for negative, 1 for positive)
        if avg >= 0.5:
            return 1
        else:
            return 0



"""
    Start training process with Bi-LSTM with attention layer and dropout layer
"""
model = BiLSTMModel(vocab_size,
                    hidden_dim, num_layers, dropout_prob)
trainer = pl.Trainer(max_epochs=50, reload_dataloaders_every_n_epochs=1,
                     enable_checkpointing=1, enable_progress_bar=1, detect_anomaly=True)
trainer.fit(model, train_dataset, val_dataset)
"""
    Save model parameters
"""
trainer.save_checkpoint('checkpoint.ckpt')
trainer.test(model, test_dataset, 'checkpoint.ckpt')
