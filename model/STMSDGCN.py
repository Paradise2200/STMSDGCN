from layers.STMSDGCN_related import *
from layers.ST_Nom_layer import *
from torch_utils.graph_process import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
class STMSDGCN(nn.Module):
    def __init__(self, num_nodes, seq_len=12,num_features=3,pred_len=12,supports=None,dropout=0.3,residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, K=3, Kt=3,**kwargs):
        super(STMSDGCN, self).__init__()
        # 更改变量名称
        length=seq_len
        in_dim=num_features
        out_dim=pred_len
        args=kwargs.get('args')
        self.args=args
        self.dropout = dropout
        self.num_nodes = num_nodes
        self.in_channels = num_features

        # # ReVIN
        # self.revin=RevIN(self.in_channels)

        self.bn = nn.BatchNorm2d(in_dim, affine=False)
        # TODO 新增的STID的embedding
        self.time_of_day_size = args.time_of_day_size
        self.day_of_week_size = args.day_of_week_size
        self.if_time_in_day = args.if_T_i_D
        self.if_day_in_week = args.if_D_i_W
        self.if_spatial = args.if_node
        self.embed_dim = args.d_model
        self.node_dim = self.embed_dim
        self.temp_dim_tid = self.embed_dim
        self.temp_dim_diw = self.embed_dim

        self.n = 0.5
        self.cluster_node = self.closest_power_of_2(num_nodes//4)
        self.node_patch = nn.Parameter(torch.empty(self.cluster_node, num_nodes))
        nn.init.xavier_uniform_(self.node_patch)

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(args.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        # temporal embeddings
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)
        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=args.num_features * args.seq_len, out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        # encoding
        self.hidden_dim = self.embed_dim + self.node_dim * \
                          int(self.if_spatial) + self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(args.num_layer)])

        self.emb_conv = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=(1, 1))
        self.in_conv = nn.Conv2d(self.hidden_dim, residual_channels, kernel_size=(1, 1))

        self.start_conv = nn.Conv2d(in_channels=self.hidden_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.nodevec1_1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
        self.nodevec1_2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)

        self.nodevec2_1 = nn.Parameter(torch.randn(self.cluster_node, 10).to(device), requires_grad=True).to(device)
        self.nodevec2_2 = nn.Parameter(torch.randn(10, self.cluster_node).to(device), requires_grad=True).to(device)

        self.supports_len = 1

        t_norm1=TNorm(num_nodes, dilation_channels)
        s_norm1=SNorm(dilation_channels)
        # length1 = (length - (2 * (3 - 1) + 1) // 1) + 1  # 根据膨胀卷积的公式计算
        self.block1 = GCNPool(dilation_channels, dilation_channels, num_nodes=num_nodes,tem_size= length - 6,Kt=3, dropout=dropout, pool_nodes=num_nodes,
                              support_len=self.supports_len,args=kwargs.get('args'),t_norm=t_norm1,s_norm=s_norm1)

        t_norm2=TNorm(self.cluster_node, dilation_channels)
        s_norm2=SNorm(dilation_channels)
        # length2 = (length1 - (2 * (2 - 1) + 1) // 1) + 1  # 根据膨胀卷积的公式计算
        self.block2 = GCNPool(dilation_channels, dilation_channels, self.cluster_node, length - 6, 3, dropout, self.cluster_node,
                              support_len=self.supports_len,args=kwargs.get('args'),t_norm=t_norm2,s_norm=s_norm2)

        self.skip_conv1 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)
        self.skip_conv2 = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1),
                                 stride=(1, 1), bias=True)

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 3),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)



    def closest_power_of_2(self, n):
        # 找到不大于 n 的最大 2 的次方
        lower = 2 ** math.floor(math.log2(n))
        # 找到大于 n 的最小 2 的次方
        upper = 2 ** math.ceil(math.log2(n))
        # 返回更接近的 2 的次方
        return lower if abs(n - lower) <= abs(n - upper) else upper


    def forward(self, input,adj,**kwargs):
        # input(B,C,N,L),return:(B,C,N,L)
        # input=self.revin(input,'norm')

        input = self.bn(input) # 在特征维度进行归一化
        x=input.clone()
        input_data = input.permute(0, 3, 2, 1)  # [B, L, N, C]
        input_data = input_data[..., range(self.in_channels)]
        seq_time = kwargs.get('seqs_time')
        pred_time = kwargs.get('targets_time')
        # time(dayofyear, dayofmonth, dayofweek, hourofday, minofhour)
        if self.if_time_in_day:
            hour = (seq_time[:, -2:-1, ...] + 0.5) * 23  # 得到第几个小时
            min = (seq_time[:, -1:, ...] + 0.5) * 59  # 得到第几分钟
            hour_index = (hour * 60 + min) / (60 / self.args.points_per_hour)
            time_in_day_emb = self.time_in_day_emb[
                hour_index[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]  # (B,N,D)
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            day = (seq_time[:, 2:3, ...] + 0.5) * (6 - 0)
            day_in_week_emb = self.day_in_week_emb[
                day[..., -1].squeeze(1).repeat(1, self.args.num_nodes).type(torch.LongTensor)]
        else:
            day_in_week_emb = None
        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        node_emb = []
        if self.if_spatial:
            # expand node embeddings
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))

        # temporal embeddings
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        # concate all embeddings
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=1)
        # encoding
        hidden = self.encoder(hidden)
        x = self.emb_conv(x) + hidden
        node_patch = torch.where(self.node_patch>self.n, 1.0, 0.0)
        x_c = torch.einsum('bcnl, wn->bcwl', x, node_patch)

        # nodes
        # A=A+self.supports[0]
        A = F.relu(torch.mm(self.nodevec1_1, self.nodevec1_2)) # 可训练邻接矩阵
        d = 1 / (torch.sum(A, -1)) # 归一化
        D = torch.diag_embed(d) # 对角化
        A1 = torch.matmul(D, A) # 得到归一化的拉普拉斯矩阵

        new_supports1 = [A1.to(device)]

        A = F.relu(torch.mm(self.nodevec2_1, self.nodevec2_2))  # 可训练邻接矩阵
        d = 1 / (torch.sum(A, -1))  # 归一化
        D = torch.diag_embed(d)  # 对角化
        A2 = torch.matmul(D, A)  # 得到归一化的拉普拉斯矩阵

        new_supports2 = [A2.to(device)]

        skip = 0
        x = self.start_conv(x) # 升特征维度
        x_c = self.start_conv(x_c)  # 升特征维度

        # S-T block1
        x = self.block1(x, new_supports1)

        s1 = self.skip_conv1(x) # 升维为了后续残差连接
        skip = s1 + skip

        # S-T block2
        x_c = self.block2(x_c, new_supports2)

        s2 = self.skip_conv2(x_c)
        s2 = torch.einsum('bcwl, wn->bcnl', s2, node_patch)
        skip = skip[:, :, :, -s2.size(3):]
        skip = s2 + skip

        # Forcing Block 类似与FeedForward
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))# F_sum1 将时间维度压成1
        x = self.end_conv_2(x) # 将特征维度降维成需要预测的时间长度
        x=x.transpose(1,3)

        # x = self.revin(x, 'denorm')
        return x  # output = [batch_size,1=dim,num_nodes,12=pred_len]
