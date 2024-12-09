import os
import yaml

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from args import args
from load_data import make_dataloader
from models.FramePredictor import FramePredictor

device = torch.device(args.device)
DATA_DIR = args.data_dir

with open('./config.yml', 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    TRAIN_EPOCHS = config['train']['epochs']
    VERBOSE_STEP = config['train']['verbose_step']
    BATCH_SIZE = config['train']['batch_size']
    BERT_EMBEDDING_DIM = config['model']['bert_embedding_dim']

bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased').to(device)


def train(data_path, tokenizer, text_extractor, net, optimizer, criterion):
    processed_file = os.path.join(data_path, 'train.pkl')
    dataloader = make_dataloader(processed_file, tokenizer, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(TRAIN_EPOCHS):
        epoch_loss = 0.0

        for i, datum in enumerate(dataloader):
            optimizer.zero_grad()

            tokens = datum['tokens'].to(device)
            token_atn_masks = datum['token_atn_masks'].to(device)
            token_lengths = datum['token_lengths'].to(device)
            text_features = {
                'embeddings': text_extractor(input_ids=tokens, attention_mask=token_atn_masks).last_hidden_state,
                'token_lengths': token_lengths
            }

            frame_lengths_hat = net(text_features)
            frame_lengths = datum['pose_atn_masks'].sum(dim=1, keepdims=True)
            loss = criterion(frame_lengths_hat, frame_lengths)

            loss.backward()
            optimizer.step()

            epoch_loss += loss

        epoch_loss /= len(dataloader)
        print(f'Epoch {epoch + 1}/{TRAIN_EPOCHS}, Loss: {epoch_loss:.4f}')


if __name__ == '__main__':
    frame_predictor = FramePredictor(BERT_EMBEDDING_DIM, 256, 0.1).to(device)
    optimizer = torch.optim.Adam(frame_predictor.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train(DATA_DIR, bert_tokenizer, bert_model, frame_predictor, optimizer, criterion)
