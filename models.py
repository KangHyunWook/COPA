#code adapted from: https://github.com/xueyunlong12589/DGCNN/blob/main/model.py
class DGCNN(nn.Module):
    def __init__(self, config):
        #in_channels(int): The feature dimension of each electrode.
        self.config = config
        in_channels= config.in_channels
        num_electrodes=config.num_electrodes
                
        k_adj = 2
        num_classes=3
        #out_channel(int): The feature dimension of  the graph after GCN.
        #num_classes(int): The number of classes to predict.
        super(DGCNN, self).__init__()
        self.K = k_adj
        out_channels=256
        self.layer1 = Chebynet(in_channels, k_adj, out_channels)
        self.BN1 = nn.BatchNorm1d(in_channels)
        self.fc = Linear(num_electrodes*out_channels, self.config.hidden_size)
        self.A = nn.Parameter(torch.FloatTensor(num_electrodes,num_electrodes).cuda(0))
        nn.init.uniform_(self.A,0.01,0.5)

    def forward(self, x):
        x=torch.squeeze(x, axis=1)

        x = self.BN1(x.transpose(1, 2)).transpose(1,2) #data can also be standardized offline

        L = normalize_A(self.A)
        result = self.layer1(x, L)
        result = result.reshape(x.shape[0], -1)
        result = self.fc(result)


        return result

class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=4, stride=2),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
        )
        
        in_features=4864
        if config.in_channels==4:
            in_features=1920
        self.fc=nn.Linear(in_features=in_features,out_features=config.hidden_size)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x=x.view(batch_size,-1)
        x=torch.unsqueeze(x, dim=1)
        
        cnn_out = self.cnn(x)
        
        cnn_out = cnn_out.view(batch_size, -1)
        final_out=self.fc(cnn_out)

        return final_out

class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        n_channels=64
        fc_size=1024
        output_size=5

        self.hidden_size=hidden_sizes=[config.lstm_hidden_size, config.hidden_size]
        rnn = nn.LSTM
        in_channels=310
        if config.model_name=='ECLGCNN':
            in_channels=170
        self.eeg_rnn1= rnn(in_channels, hidden_sizes[0], bidirectional=True)
        self.eeg_rnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        self.layer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        
        self.project_eeg=nn.Sequential()
        self.project_eeg.add_module('project_eeg', nn.Linear(in_features=4*hidden_sizes[0],out_features=hidden_sizes[1]))


        
    def extract_features(self, x, batch_size):
        x=torch.unsqueeze(x,dim=0)
        
        packed_h1, (h1, _) = self.eeg_rnn1(x)
        normed_h1 = self.layer_norm(packed_h1)
        
        _, (h2, _)=self.eeg_rnn2(normed_h1)


        o=torch.cat((h1, h2), dim=2).permute(1,0,2).contiguous().view(batch_size, -1)    
        o=self.project_eeg(o)


        return o
    
    
    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []

        x=x.reshape(batch_size, -1)
        
        o = self.extract_features(x, batch_size)


        return o

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x=torch.unsqueeze(x,dim=3)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        out=torch.squeeze(x,dim=3)
        return out


class COPA(nn.Module):
    def __init__(self, config):
        super(COPA, self).__init__()
        self.config=config 
        self.dropout_rate = dropout_rate = config.dropout

        self.dgcnn=DGCNN(config)
        self.cnn = CNN(config)
        self.lstm = LSTM(config)
        self.activation = self.config.activation()
        
        self.project_private_ccf = nn.Sequential()
        self.project_private_ccf.add_module('project_private_ccf_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.project_private_ccf.add_module('project_private_ccf_activation_1', nn.Sigmoid())

        self.project_private_spatial = nn.Sequential()
        self.project_private_spatial.add_module('project_private_spatial_1',nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.project_private_spatial.add_module('project_private_spatial_activation_1', nn.Sigmoid())

        self.project_private_temporal = nn.Sequential()
        self.project_private_temporal.add_module('project_private_temporal_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.project_private_temporal.add_module('project_private_temporal_activation_1', nn.Sigmoid())

        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        self.p_recon_con = nn.Sequential()
        self.p_recon_con.add_module('recon_con_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.p_recon_loc = nn.Sequential()
        self.p_recon_loc.add_module('recon_loc_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.p_recon_seq = nn.Sequential()
        self.p_recon_seq.add_module('recon_seq_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)) 

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= config.n_classes))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        encoder_layer_feat = nn.TransformerEncoderLayer(d_model=310, nhead=2)
        self.transformer_encoder_feat = nn.TransformerEncoder(encoder_layer_feat, num_layers=1)
        
        self.glu=GLU(62,0.01)
    
    def alignment(self, con, loc, seq):
        batch_size = con.shape[0]

        self.shared_private(con, loc, seq)
        
        self.reconstruct()
        
        h = torch.stack((self.private_con, self.private_loc, self.private_seq, self.shared_con, self.shared_loc,  self.shared_seq), dim=0)
        h = self.transformer_encoder(h)  
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)

        o = self.fusion(h)

      return o

    
    def reconstruct(self,):
        self.con = (self.private_con + self.shared_con)
        self.loc = (self.private_loc + self.shared_loc)
        self.seq = (self.private_seq + self.shared_seq)

        self.con_recon = self.p_recon_con(self.con)
        self.loc_recon = self.p_recon_loc(self.loc)
        self.seq_recon = self.p_recon_seq(self.seq)

    
    def shared_private(self, con, loc, seq):
        self.private_con = self.project_private_con(con)
        self.private_loc = self.project_private_loc(loc)
        self.private_seq = self.project_private_seq(seq)

        self.shared_con = self.shared(con)
        self.shared_loc = self.shared(loc)
        self.shared_seq = self.shared(seq)
        
    def forward(self,x):
        x=self.glu(x)
        batch_size=x.shape[0]
        con=self.dgcnn(x) # EEG channel connectivity features      
        loc = self.cnn(x)
        seq = self.lstm(x)

        o=self.alignment(con, loc, seq)
        
        return o 
