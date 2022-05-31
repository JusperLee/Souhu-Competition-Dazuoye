import torch
import torch.nn as nn
import torch.nn.functional as F

class rec_net(nn.Module):
    def __init__(self, in_dim):
        super(rec_net, self).__init__()
        self.act1 = F.relu
        self.act2 = F.sigmoid
        self.fc1 = nn.Linear(in_dim, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.act2(self.fc3(x))
        return x

class rec_net_embedding(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.embedding = rec_embedding()
        self.act1 = F.relu
        self.act2 = F.softmax
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, user, feed, city):
        x = self.embedding(user, feed, city)
        x = self.act1(self.fc1(x))
        x = self.act1(self.fc2(x))
        x = self.fc3(x)
        return x

class rec_net_emotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = rec_embedding()
        self.item_emb_fc = nn.Sequential(
            # nn.Linear(768*4+4, 200),
            nn.Linear(768*6+4, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32*2+64*2+256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.norm_x = nn.LayerNorm(32*2+64*2)
        self.simple_feature_fc = nn.Linear(1, 2)

    def forward(self, user, feed, city, item_ebds, item_emo_feature, sequence_embedding, item_emb_seq):
        item_feature = self.item_emb_fc(torch.cat([item_ebds.to(torch.float32), 
                                        item_emo_feature.to(torch.float32), sequence_embedding.to(torch.float32)], dim=1))
        x = self.embedding(user, feed, city, item_emb_seq)
        item_emb = x[:, 32:32+64]
        item_seq_emb = x[:, 32*2+64:32*2+64*2]
        item_sim = torch.sum(item_emb * item_seq_emb, dim=1) / (torch.norm(item_emb, dim=1)* \
            torch.norm(item_seq_emb, dim=1))

        sim_out = self.simple_feature_fc(item_sim.unsqueeze(1))
        # x = self.norm_x(x)
        x = torch.cat([x, item_feature], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x += sim_out
        return x

class rec_embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Embedding(300000, 32)
        self.feed_embedding = nn.Embedding(4000, 64)
        self.city_embedding = nn.Embedding(340, 32)
    
    def forward(self, user, feed, city, item_emb_seq):
        user_out = self.user_embedding(user)
        feed_out = self.feed_embedding(feed)
        city_out = self.city_embedding(city)
        # item_emb_seq: shape: (batch, max_length of seq)
        # the padding for no item in sequence is -1
        item_seq_mask = torch.ones_like(item_emb_seq)
        item_seq_mask[item_emb_seq == -1] = 0
        item_seq_mask = item_seq_mask.unsqueeze(-1)
        item_emb_seq[item_emb_seq == -1] = 0
        item_seq_embedding = self.feed_embedding(item_emb_seq)
        item_seq_embedding = item_seq_mask * item_seq_embedding
        item_seq_embedding = item_seq_embedding.mean(dim=1)
        return torch.cat([user_out, feed_out, city_out, item_seq_embedding], dim=1)

class rec_net_emotion_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = rec_embedding_old()
        self.item_emb_fc = nn.Sequential(
            nn.Linear(768*6+4, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(32*3+256+100, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)
        self.norm_x = nn.LayerNorm(32*3)
        self.simple_feature_fc = nn.Linear(1, 2)

    def forward(self, user, feed, city, item_ebds, item_emo_feature, sequence_embedding, item_emb_seq, item2vec):
        item_feature = self.item_emb_fc(torch.cat([item_ebds.to(torch.float32), 
                                    item_emo_feature.to(torch.float32), sequence_embedding.to(torch.float32)], dim=1))
        x = self.embedding(user, feed, city)
        x = self.norm_x(x)
        x = torch.cat([x, item_feature, item2vec], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class rec_embedding_old(nn.Module):
    def __init__(self):
        super().__init__()
        self.user_embedding = nn.Embedding(300000, 32)
        self.feed_embedding = nn.Embedding(4000, 32)
        self.city_embedding = nn.Embedding(340, 32)
    
    def forward(self, user, feed, city):
        user_out = self.user_embedding(user)
        feed_out = self.feed_embedding(feed)
        city_out = self.city_embedding(city)
        return torch.cat([user_out, feed_out, city_out], dim=1)
