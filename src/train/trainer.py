import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torchaudio.transforms as T

import lightning as L
import transformers
from tslearn.metrics import  dtw_path_from_metric

from models import ModelArgs, BiMambaEncoder
from vector_quantize_pytorch import VectorQuantize

class AudioTokenizer(L.LightningModule):
    def __init__(self, 
                 config):
        super().__init__()

        self.config = config
        mamba_args =ModelArgs(
            d_model=self.config['d_model'],
            d_project=self.config['d_project'],
            d_state=self.config['d_state'],
            d_conv=self.config['d_conv'],
            expand=self.config['expand'],
            n_layer=self.config['n_layer'],
            headdim=self.config['headdim'],
            block_type=self.config['block_type'],
            )
        self.encoder = BiMambaEncoder(mamba_args)

        self.vq_layer = VectorQuantize(
                                dim = self.config['codeword_dims'],
                                codebook_size = self.config['num_codewords'],     # codebook size
                                decay = self.config['ema_decay'],      
                                commitment_weight = self.config['beta'],   # the weight on the commitment loss
                                kmeans_init = True,   # set to True
                                kmeans_iters = 10,  
                                use_cosine_sim = True,
                                threshold_ema_dead_code = 2,
                                orthogonal_reg_weight = self.config['ortho_weight'],                 # in paper, they recommended a value of 10
                                orthogonal_reg_active_codes_only = False)
        

        # self.featextract = torch.nn.Sequential(T.MelSpectrogram(sample_rate=self.config['fs'],
        #                                                         n_fft=int(self.config['win_size'] * self.config['fs']),
        #                                                         hop_length=int(self.config['win_hop'] * self.config['fs']),
        #                                                         n_mels=self.config['n_mels'],
        #                                                         center=True),
        #                                     T.AmplitudeToDB())


        self.temperature=self.config['temperature']
        self.cost_factors=self.config['cost_factors']
        self.optim=self.config['optimizer']
        self.lr=self.config['lr']
        self.wt_decay=self.config['wt_decay']
        self.lr_scheduler=self.config['lr_scheduler']
        self.add_augment=self.config["add_augment"]

        self.save_hyperparameters()

    # def forward(self, x): 
    #     # this is actually called by default in the predict_step
    #     # but make sure to use .eval() and no_grad() context manager
    #     # you can do inference with this method for research purpose
    #     return self.model(x)

    def _shared_step(self, batch, mode):
        log_values = {}
        
        bsz = len(batch['utterance_idx'][0])
        anc = batch['utterances'][0]
        pos = batch['utterances'][1]

        z_ctx_anc = self.encoder(anc)
        z_ctx_pos = self.encoder(pos)

        z_ctx_anc_quant, indices_anc, codebook_loss_anc = self.vq_layer(z_ctx_anc) 
        z_ctx_pos_quant, indices_pos, codebook_loss_pos = self.vq_layer(z_ctx_pos) 

        
        if self.add_augment:
            noisy_pos = batch['utterances'][2]
            z_ctx_noisy_pos = self.encoder(noisy_pos)
            z_ctx_noisy_pos_quant, indices_noisy_pos, codebook_loss_noisy_pos = self.vq_layer(z_ctx_noisy_pos)

        
        l1 = 0.0
        loss = 0
        tot_frames = 0
        for idx in range(bsz):
            n_idx = self._get_neg_sample_idx(bsz, idx)
            if idx < 30 and mode == "train":
                mel_a = anc[idx][~batch['utterance_idx'][0][idx].bool(),:]
                mel_p = pos[idx][~batch['utterance_idx'][1][idx].bool(),:]
                mel_n = pos[n_idx]#[~batch['utterance_idx'][1][n_idx].bool(),:]
                triplets = self._get_triplet_indices(mel_a, mel_p, mel_n[0])


                aa = z_ctx_anc[idx][~batch['utterance_idx'][0][idx].bool(),:]
                if self.add_augment:
                    pp = z_ctx_noisy_pos[idx][~batch['utterance_idx'][1][idx].bool(),:]
                    nn = z_ctx_noisy_pos[n_idx].squeeze()
                else:
                    pp = z_ctx_pos[idx][~batch['utterance_idx'][1][idx].bool(),:]
                    nn = z_ctx_pos[n_idx].squeeze()#[~batch['utterance_idx'][1][n_idx].bool(),:]
                
                                
                sim_pos = torch.einsum('nd, nd -> n',  aa[triplets[:,0]], pp[triplets[:,1]])
                sim_neg = torch.einsum('nd, nkd -> nk',  aa[triplets[:,0]], nn[triplets[:,2:]])
                sim = torch.cat([sim_pos.unsqueeze(1), sim_neg], dim=1)/self.temperature
                l1 += F.cross_entropy(sim, torch.zeros((sim.size(0),), dtype=torch.long, device=sim.device), reduction="sum")
                tot_frames += len(triplets)

        # codebook loss
        log_values[mode+"_codebook_loss"] = 0.5*(codebook_loss_anc + codebook_loss_pos)*self.cost_factors['vq_loss']
        loss = loss + log_values[mode+"_codebook_loss"] 

        # cosine sim-based contrastive loss on frame level
        if mode == "train":
            log_values[mode+'frames'] = tot_frames
            log_values[mode+'_contra_loss'] = self.cost_factors["triplet_loss"]*(l1/tot_frames)
            loss = loss + log_values[mode+'_contra_loss']

        # for logging purposes
        self.z_anc = z_ctx_anc[0]   
        self.z_pos = z_ctx_pos[0]
        self.z_anc_quant = z_ctx_anc_quant[0]
        self.z_pos_quant = z_ctx_pos_quant[0]
        self.word = batch['word'][0]        
        codes_anc = indices_anc
        codes_pos = indices_pos
        if self.add_augment:
            codes_noisy_pos = indices_noisy_pos
            print(len(torch.unique(codes_anc)), len(torch.unique(codes_pos)),len(torch.unique(codes_noisy_pos)) )
            print(batch['word'][0], codes_anc[0][~batch['utterance_idx'][0][0].bool()], codes_pos[0][~batch['utterance_idx'][1][0].bool()], codes_noisy_pos[0][~batch['utterance_idx'][1][0].bool()])
            acc = self._get_word_accuracy(codes_anc, codes_noisy_pos, batch['utterance_idx'])
            log_values[mode+"_acc_noise"] = acc
        else:
            print(len(torch.unique(codes_anc)), len(torch.unique(codes_pos)))
            print(batch['word'][0], codes_anc[0][~batch['utterance_idx'][0][0].bool()], codes_pos[0][~batch['utterance_idx'][1][0].bool()])

        acc = self._get_word_accuracy(codes_anc, codes_pos, batch['utterance_idx'])
        log_values[mode+"_acc"] = acc
        
        self.log_dict(log_values, on_epoch=True, on_step=True, logger=True, prog_bar=True)
        return {'loss': loss, 'z': z_ctx_anc}
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="valid")
    
    def predict_step(self, batch, batch_idx):
        z_ctx = self.encoder(batch)
        z_ctx_quant, indices, _ = self.vq_layer(z_ctx)
        return z_ctx, z_ctx_quant, indices

    def configure_optimizers(self):
        if self.optim == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optim == "adamw":
            assert self.wt_decay >0, "weight decay value must be non zero"
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wt_decay)
        else:
            raise NotImplementedError
        
        if self.lr_scheduler:
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, 
                                                         num_warmup_steps=10, 
                                                         num_training_steps=300)
            # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=5e-4)
            return {"optimizer": optimizer,  'lr_scheduler':{"scheduler": scheduler, "interval":"epoch"}}
        else:
            return {"optimizer": optimizer}

    @staticmethod
    def _cosine_distance(x:torch.Tensor,y:torch.Tensor)->float:
        assert x.dim() == y.dim() == 3, f"the inputs must be 3 dimensional (batch, seq_len, feat_dims) tensor"
        distance =  1 - torch.einsum('bqd, bkd -> bqk', [x,y])
        return distance
    
    def _get_neg_sample_idx(self, N, ignore_idx):
        idx = torch.randint(N,(1,))
        if idx == ignore_idx:
            return self._get_neg_sample_idx(N, ignore_idx)
        return idx

    @staticmethod
    def _get_triplet_indices(a,p,n):
        a = a/torch.linalg.norm(a, dim=1).reshape(-1,1)
        p = p/torch.linalg.norm(p, dim=1).reshape(-1,1)
        path, cost = dtw_path_from_metric(a.detach().cpu().numpy(), p.detach().cpu().numpy(), metric="cosine")

        # get alignment mapping
        alignment = {}
        for tupl in path:
            if tupl[0] not in alignment:
                alignment[tupl[0]] = []
            alignment[tupl[0]].append(tupl[1])
        
        # get triplet indices
        indices = []
        for idx in alignment.keys():
            # sample a positive from other utternace
            sim_score = a[idx].unsqueeze(0) @ p[alignment[idx]].T
            pos_idx = alignment[idx][torch.argmax(sim_score.squeeze_())] # pick frames which have max similarity
            neg_idx = np.random.choice(len(n), 30) #randomly pick one frame from other word utterance
            i = [idx, pos_idx]
            i.extend(neg_idx.tolist())
            indices.append(i)
        return torch.tensor(indices)

    @staticmethod
    def _get_word_accuracy(codes_anc, codes_pos, utterance_mask):
        acc = []
        for idx in range(len(codes_anc)):
            indices_anc = codes_anc[idx][~utterance_mask[0][idx].bool()]
            indices_pos = codes_pos[idx][~utterance_mask[1][idx].bool()]
            match_prop = len(set(indices_anc.cpu().numpy()).intersection(set(indices_pos.cpu().numpy())))/len(set(indices_anc.cpu().numpy()).union(set(indices_pos.cpu().numpy())))
            acc.append(match_prop)
        return np.mean(acc)
    
    @classmethod
    def load_from_pretrained(cls, checkpoint, gpu_index):
        model = cls.load_from_checkpoint(checkpoint, map_location=torch.device("cuda:"+str(gpu_index)))
        model.eval()
        return model
    
    def extract_tokens(self, audio_data, hop, gpu_index="0"):

        featextract = torch.nn.Sequential(T.MelSpectrogram(sample_rate=self.config['fs'],
                                                                n_fft=int(self.config['win_size'] * self.config['fs']),
                                                                hop_length=int(self.config['win_hop'] * self.config['fs']),
                                                                n_mels=self.config['n_mels'],
                                                                center=True).to(self.device),
                                            T.AmplitudeToDB().to(self.device))

        if gpu_index is None:
            raise RuntimeError("The inference need GPU. Specify GPU index to use")
        else:
            device = "cuda:"+gpu_index

        chunk_sz = int(self.config['seglen']*self.config['fs'])
        hop_sz = int(hop*self.config['fs']) 

        audio_chunks = [audio_data[start_idx: start_idx+chunk_sz] for start_idx in np.arange(0,len(audio_data)-chunk_sz+1, hop_sz)]  

        audio_chunks = torch.stack(audio_chunks,dim=0)
        audio_feats = featextract(audio_chunks.to(device)).permute(0,2,1)
        
        _, _, indices = self.get_predictions(audio_feats, device) #self.predict_step(audio_feats.cuda(), 1)
        # torch.cuda.empty_cache()
        # time_stamps = torch.arange(0,len(audio_chunks))*hop
        # assert len(time_stamps) == len(indices), f"wrong shape {len(time_stamps),len(indices)}"
        # tokens_with_timestamp = torch.concatenate((indices.detach().cpu(), time_stamps.reshape(-1,1)), dim=1)
        # return tokens_with_timestamp
        return indices.detach().cpu().type(torch.int16)
    

    def get_predictions(self, feats_utt, device):

        # Long audio track --> large batch size --> GPU out of memory issue
        if len(feats_utt) >256:
            Z_CTX = []
            Z_CTX_QUANT = []
            INDICES = []
            steps = int(len(feats_utt)/256)
            remainder = int(len(feats_utt)/256)
            for i in range(steps):
                z_ctx, z_ctx_quant, indices = self.predict_step(feats_utt[i*256: (i+1)*256].to(device), 1)
                Z_CTX.append(z_ctx.detach().cpu())
                Z_CTX_QUANT.append(z_ctx_quant.detach().cpu())
                INDICES.append(indices.detach().cpu())
            if remainder > 0:
                z_ctx, z_ctx_quant, indices = self.predict_step(feats_utt[steps*256:].to(device), 1) 
                Z_CTX.append(z_ctx.detach().cpu())
                Z_CTX_QUANT.append(z_ctx_quant.detach().cpu())
                INDICES.append(indices.detach().cpu())
            z_ctx = torch.cat(Z_CTX)
            z_ctx_quant = torch.cat(Z_CTX_QUANT)
            indices = torch.cat(INDICES)
        else:
            z_ctx, z_ctx_quant, indices = self.predict_step(feats_utt.to(device), 1)

        return z_ctx, z_ctx_quant, indices
















   
