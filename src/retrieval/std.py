import sys
sys.path.append("../")

import os
import glob
import math
import torch
import pickle
import argparse
import numpy as np
import torch.nn.functional as F
import IPython.display as ipd

from train import AudioTokenizer
from utils.audio_utils import read_audio
from utils.metrics import average_precision, MTWV

import matplotlib.pyplot as plt
from tqdm import tqdm

from typing import Optional, Tuple, List, Dict


# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

class STD:
    def __init__(self,
                 audio_path: str,
                 checkpoint: str,
                 fs: Optional[int]=16000,
                 gpu_index: Optional[int]=0,
                 ):
        
        self.audio_path = audio_path
        self.fs = fs
        self.model = AudioTokenizer.load_from_pretrained(checkpoint, gpu_index)
        self.gpu_index = str(gpu_index)
        print(self.model.device)
        # self.model.to("cuda:"+self.gpu_index)

    def extract_query(self, fname: str,
                    time_bd: Tuple,
                    pad_mode: Optional[str]="context",
                    inp_len: float=1.0,
                    total_frames: int=101,
                    token_interval: float=0.01) -> List[torch.Tensor]:
    
        audio_data = read_audio(fname)
        inp_len = int(self.fs*inp_len)
        token_interval = int(self.fs*token_interval)

        # pad width
        utt_bound = (np.array(time_bd)*self.fs).astype(int)
        utt_len = int(utt_bound[1]-utt_bound[0])
        if utt_len % 2 > 0:
            utt_bound[1] = utt_bound[1]+1
            utt_len+=1
        utt_pad_len = int((inp_len - utt_len)/2)

        if pad_mode == "context":
            # pad with context
            utt_zero_pad_l, utt_zero_pad_r = max(0,math.floor(utt_pad_len)-utt_bound[0]), max(0,math.ceil(utt_pad_len)-(len(audio_data)-utt_bound[1]))
            utt_data = audio_data[max(0,utt_bound[0]-math.floor(utt_pad_len)) : min(len(audio_data), utt_bound[1]+math.ceil(utt_pad_len))]
            utt_data = F.pad(utt_data, (utt_zero_pad_l, utt_zero_pad_r))
        elif pad_mode == "constant":
            # zero padding
            audio_data = audio_data[utt_bound[0]:utt_bound[1]]
            utt_data = F.pad(audio_data, (utt_pad_len, utt_pad_len))
        else:
            raise RuntimeError("wrong pad_mode specified. Available options: context or constant")

        # get pad mask
        pad_frames_len = math.ceil((utt_pad_len/token_interval))
        mask = torch.zeros(total_frames, dtype=torch.int8)
        mask[:pad_frames_len] = 1
        mask[-pad_frames_len:] = 1

        assert len(utt_data) == inp_len, f"expected utterance length mismatched, got {len(utt_data)} and expected: {inp_len}"
        return utt_data, mask

    def tokenize(self,
                audio_data: torch.Tensor,
                win_hop: Optional[float]=0.01) -> torch.Tensor: 
        
        if audio_data.dim() > 1:
            audio_data = audio_data.reshape(-1)
        audio_tokens = self.model.extract_tokens(audio_data, hop=win_hop, gpu_index=self.gpu_index) 
        # audio_tokens = audio_tokens.reshape(-1) #################
        return audio_tokens

    def add(self,
            filenames: List,
            dbase: Optional[str]=None,
            win_hop: Optional[float]=0.01) -> Dict:
        
        if dbase is None:
            dbase = {}
        else:
            print(f"adding to pre-exisiting database located at: {dbase}")
            dbase = pickle.load(open(dbase, "rb"))
            
        for fname in tqdm(filenames):
            try:
                audio_data = read_audio(fname)
                transcript = self.tokenize(audio_data,  win_hop=win_hop)
                dbase[fname] = transcript
            except Exception as e:
                print(e)
        return dbase
    
    def index(self,
            dbase: str,
            sample_rate: Optional[int]=1):
    
        self.model = None
        if isinstance(dbase, str):
            dbase = pickle.load(open(dbase, "rb"))

        inverted_index = {}
        for fname in tqdm(list(dbase.keys())):
            tokens = dbase[fname]
            # tokens = tokens.reshape(-1,101) ################

            if sample_rate > 1:
                tokens = tokens[::sample_rate]
            
            for frame_idx, frame_tokens in enumerate(tokens):
                frame_bigrams =  set(self.bigram(frame_tokens)) #  set(tuple(zip(frame_tokens[:-1].tolist(), frame_tokens[1:].tolist())))
                
                for bigram in frame_bigrams:
                    if bigram not in inverted_index:
                        inverted_index[bigram] = {}

                    if fname not in inverted_index[bigram]:
                        inverted_index[bigram][fname] = []
                        
                    inverted_index[bigram][fname].append(frame_idx)
        return inverted_index
    
    @staticmethod
    def jaccard_similarity(indices_anc, indices_pos):

        if isinstance(indices_anc, torch.Tensor):
            indices_anc = indices_anc.tolist()
            indices_pos = indices_pos.tolist()
            
        # computes intersection over union
        unique_indices_anc = set(indices_anc)
        unique_indices_pos = set(indices_pos)
        intersect = unique_indices_anc.intersection(unique_indices_pos)
        union = unique_indices_anc.union(unique_indices_pos)
        sim = len(intersect)/len(union)
        return sim
    
    @staticmethod
    def deduplicate(seq: List) -> List:
            ddp_t = [seq[0]]
            for char in seq[1:]:
                if char == ddp_t[-1]:
                    continue
                else:
                    ddp_t.append(char)
            return ddp_t
    
    @staticmethod
    def bigram(x):
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        return tuple(zip(x[:-1], x[1:]))

    def naive_search(self,
                query:torch.Tensor,
                dbase: Dict,
                sample_rate: Optional[int]=1,
                deduplicate_tokens: Optional[bool]=False,
                bigram: Optional[bool]=False,
                )->Dict:
        

        if isinstance(query, torch.Tensor):
            query = query.tolist()

        if deduplicate_tokens:
            query = self.deduplicate(query)

        if bigram:
            query = self.bigram(query) 
            

        results = {}
        for rfname, ref_audio in dbase.items():
            
            if ref_audio.dim()==1:
                ref_audio = ref_audio.reshape(-1, 101)

            if sample_rate > 1:
                ref_audio = ref_audio[::sample_rate]

            max_sim = 0
            for frame_idx, frame_token in enumerate(ref_audio):
                for start_idx in range(len(frame_token)-len(query)-1):
                    seg = frame_token[start_idx:start_idx+len(query)+1].tolist()
                    
                    if deduplicate_tokens:
                        seg = self.deduplicate(seg)
                    
                    if bigram:
                        seg = self.bigram(seg)
                    
                    frame_sim =  self.jaccard_similarity(seg, query)

                    if frame_sim >= max_sim:
                        max_sim = frame_sim

            results[rfname] = max_sim

        
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
            
        return results

    def search(self,
                query_metadata: List,
                dbase: Dict,
                index: Dict,
                hop: Optional[float]= 0.01,
                sample_rate: Optional[int] = 1,
                query_pad_mode: Optional[str] = "context"
                ):
        
        query_fname, query_time_bd = query_metadata
        query_fname = os.path.join(self.audio_path, query_fname)
        query, query_mask = self.extract_query(query_fname, query_time_bd, pad_mode=query_pad_mode)
        query = self.tokenize(query).reshape(-1) ####################
        query = query[~query_mask.bool()]


        # collect all candidate frames across reference tracks
        query = self.bigram(query)
        cands = {}
        for qbigram in set(query):
            for fname, frames in index[qbigram].items():
                if fname not in cands:
                    cands[fname] = frames
                else:
                    cands[fname].extend(frames)
        
        print(f"total frame candidates: {len(cands)}")
        results = {}
        # loop over all candidate audio tracks
        for fname, frames in tqdm(cands.items()):
            transcript = dbase[fname]#.reshape(-1,101) #################
            if sample_rate > 1:
                transcript = transcript[::sample_rate]

            max_sim =0
            max_sim_frame_time = 0
            # analyze each candidate frames within a candidate audio track
            for frame_idx in frames:
                frame_tokens = transcript[frame_idx]
                
                # find subsequence within a candidate frame that best matches the query
                for start_idx in range(len(frame_tokens)-len(query)-1):
                    seg = frame_tokens[start_idx:start_idx+len(query)+1]
                    seg = self.bigram(seg)
                    seg_sim = self.jaccard_similarity(seg, query)

                    if seg_sim >= max_sim:
                        max_sim = seg_sim
                        max_sim_frame_time = frame_idx*sample_rate*hop
                        if max_sim > 0.9:
                            break
            
            results[fname] = (max_sim, max_sim_frame_time)
            
        results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        return results    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required= True, choices=["add", "index", "search"])
    parser.add_argument("--fname", type=str, required=False)
    parser.add_argument("--dbase", type=str, required=False)
    parser.add_argument("--index", type=str, required=False)
    parser.add_argument("--query_path", type=str, required=False)
    parser.add_argument("--sample_rate", type=int, required=False, default=1)
    args = parser.parse_args()



    # model-specific parameters

    # # 128
    # dbase_dir = "/home/anup/STD/data/dbase/Ours-train-360/LS-test-clean-100/128"
    # checkpoint = "/home/anup/STD/checkpoints/LS-clean-360_new/codes128_1s_LS360/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=504-valid_loss=0.00-train_loss=0.00.ckpt"

    # # 256
    # dbase_dir = "/home/anup/STD/data/dbase/Ours-train-360/LS-test-clean-100/256"
    # checkpoint = "/home/anup/STD/checkpoints/LS-clean-360_new/codes256_1s_LS360/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=1005-valid_loss=0.00-train_loss=0.00.ckpt"

    # 512
    dbase_dir = "/home/anup/STD/data/dbase/Ours-train-360/LS-test-clean-100/512"
    checkpoint = "/home/anup/STD/checkpoints/LS-clean-360_new/codes512_1s_LS360/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=781-valid_loss=0.00-train_loss=0.00.ckpt"

    # # 1024
    # dbase_dir = "/home/anup/STD/data/dbase/Ours-train-360/LS-test-clean-100/1024"
    # checkpoint = "/home/anup/STD/checkpoints/LS-clean-360_new/codes1024_1s_LS360/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=539-valid_loss=0.00-train_loss=0.00.ckpt"

    # TRANSFORMER
    # # 512
    # dbase_dir = "/home/anup/STD/data/dbase/Transformers-train-360/LS-test-clean-100/512"
    # checkpoint = "/home/anup/STD/checkpoints/transformers_codes512_1s_LS100/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=775-valid_loss=0.00-train_loss=0.00.ckpt"

    #MAMBA
    # #512
    # dbase_dir = "/home/anup/STD/data/dbase/Mamba-train-360/LS-test-clean-100/512"
    # checkpoint = "/home/anup/STD/checkpoints/unimamba_codes512_1s_LS100/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=617-valid_loss=0.00-train_loss=0.00.ckpt"

    # #1024
    # dbase_dir = "/home/anup/STD/data/dbase/Mamba-train-360/LS-test-clean-100/1024"
    # checkpoint = "/home/anup/STD/checkpoints/unimamba_codes1024_1s_LS100/bsz:128_lr:0.0005_seg:1.0/checkpoints/epoch=614-valid_loss=0.00-train_loss=0.00.ckpt"



    # general parameters
    audio_path= "/DATA/datasets/anup_exp_data_dump/datasets/LibriSpeech/"
    subset = "train-clean-100"
    gpu_index = 0


    api = STD(audio_path=audio_path ,
          checkpoint=checkpoint,
          gpu_index=gpu_index,
          )
    print("STD API ready to use...")
    

    if args.task == "add":
        rfnames = glob.glob(os.path.join(audio_path, subset, "**/*.flac"), recursive=True)
        dbase = api.add(rfnames, win_hop=0.01)
        pickle.dump(dbase, open(os.path.join(dbase_dir, args.fname), "wb")) #"LS-100_dbase.pkl"
    
    if args.task == "index":
        if args.dbase is None:
            raise ValueError("database path not specified")
        dbase = os.path.join(dbase_dir,args.dbase)
        index = api.index(dbase, sample_rate=args.sample_rate)
        pickle.dump(index, open(os.path.join(dbase_dir, args.fname), "wb"))
    
    if args.task == "search":
        assert str(args.sample_rate) in args.index, "sample_rate mismatched"
        R = {}
        # R = pickle.load(open("/home/anup/STD/data/dbase/Ours-train-360/LS-test-clean-100/512/results_sample_rate3_type2.pkl", "rb"))
        dbase = pickle.load(open(os.path.join(dbase_dir, args.dbase), "rb"))
        index = pickle.load(open(os.path.join(dbase_dir, args.index), "rb"))
        print("index loaded...")
        queries = pickle.load(open(args.query_path, "rb"))

        for query_word in tqdm(list(queries.keys())):
            if query_word not in R:
                print(f"searching spoken term: {query_word}")
                results = api.search(queries[query_word], dbase, index, args.sample_rate)
                R[query_word] = results
                pickle.dump(R, open(os.path.join(dbase_dir ,args.fname), "wb"))
            else:
                print(f"{query_word} already exists")
