import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os

class Item2VecDataset(Dataset):
    def __init__(self, data_path):
        number = 0
        with tqdm(total = os.path.getsize(data_path),desc="load training dataset") as pbar:
            with open(data_path,"r") as f:
                # 获得训练数据的总行数
                for line in f:
                    number+=1
                    pbar.update(len(line))
        self.number = number
        self.fopen = open(data_path,'r')
    def __len__(self):
        return self.number
    def __getitem__(self, index):
        line = self.fopen.__next__()
        line_list = line.strip().split("\t")
        label = int(line_list[2])
        xi = int(line_list[0])
        xj = int(line_list[1])
        label = torch.tensor(label, dtype=torch.float32)
        xi = torch.tensor(xi, dtype=torch.long)
        xj = torch.tensor(xj, dtype=torch.long)

        return xi, xj, label

class Item2Vec(nn.Module):
    def __init__(self, args):
        super(Item2Vec, self).__init__()
        self.shared_embedding = nn.Embedding(args['vocabulary_size'], args['embedding_dim'])
        self.lr = args['learning_rate']
        self.epochs = args['epochs']
        self.out_act = nn.Sigmoid()
        self.data_path = args['data_path']
        self.batch_size = args['batch_size']

    def forward(self, target_i, context_j):
        target_emb = self.shared_embedding(target_i) # batch_size * embedding_size
        context_emb = self.shared_embedding(context_j) # batch_size * embedding_size
        output = torch.sum(target_emb * context_emb, dim=1)
        output = self.out_act(output)

        return output.view(-1)

    def fit(self):
        if torch.cuda.is_available():
            self.cuda()
        else:
            self.cpu()
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

        last_loss = 0.
        for epoch in range(1, self.epochs + 1):
            self.train()
            current_loss = 0.
            item2vec_dataset = Item2VecDataset(self.data_path)
            train_loader = DataLoader(item2vec_dataset, batch_size=self.batch_size, shuffle=False,num_workers=1)
            for target_i, context_j, label in tqdm(train_loader,desc=f"epoch:{epoch}"):
                if torch.cuda.is_available():
                    target_i = target_i.cuda()
                    context_j = context_j.cuda()
                    label = label.cuda()
                else:
                    target_i = target_i.cpu()
                    context_j = context_j.cpu()
                    label = label.cpu()
                self.zero_grad()
                prediction = self.forward(target_i, context_j)
                loss = criterion(prediction, label)
                if torch.isnan(loss):
                    raise ValueError(f'Loss=Nan or Infinity: current settings does not fit the model')
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
            torch.save(self.shared_embedding.weight, f'item_embedding_146790x100_epoch{epoch}.pt')
            print(f'[Epoch {epoch:03d}] - training loss={current_loss:.4f}')
            delta_loss = float(current_loss - last_loss)
            if (abs(delta_loss) < 1e-5) and self.early_stop:
                print('Satisfy early stop mechanism')
                break
            else:
                last_loss = current_loss

    def predict(self):
        pass