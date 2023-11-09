import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size

        device = torch.device(device)
        # 给模块添加缓冲区，默认同参数一起保存
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().to(device))

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper

        emb_i.shape=b,feature (32*512)
        """
        # L2范数归一化，结果的余弦相似度和欧氏距离等价，只有系数和幂次的差距
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        # 64*512
        representations = torch.cat([z_i, z_j], dim=0)
        # 64*64,mij，ai与bj行的余弦相似度
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        # sim_ij=sim_ji
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        # 对应元素相乘，用于将对角线的1归零
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        # 按行求和
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        # 自己和自己的另一种变化最像，与同batch即不同时空的特征表示最不像
        return loss
