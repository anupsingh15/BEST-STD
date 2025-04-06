import torch
import numpy as np
import lightning.pytorch as L
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tslearn.metrics import  dtw_path_from_metric
import scipy.spatial.distance as dist


class MyCallBack(L.Callback):
    def __init__(self):
        super().__init__()

    def plot_img(self, figure,figname,logger,trainer):
        figure.tight_layout()
        figure.canvas.draw()  #dump to memory
        img = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(figure.canvas.get_width_height()[::-1] + (3,)) #read from memory buffer
        img = img / 255.0 #RGB
        logger.experiment.add_image(figname, img, global_step=trainer.global_step, dataformats='HWC') # add to logger
        plt.close(figure)
        return

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx): 
        if batch_idx % 20 == 0:
            logger = trainer.logger

            # log codewords usage
           
            #     ideal_prob = 1/pl_module.kmeans_quantizer.num_codewords
            #     density, bins = torch.histogram(pl_module.kmeans_quantizer.probs.detach().cpu(), bins=250, density=True)
            #     # ax.bar(bins[:-1], density, width=0.01, edgecolor="b")
            #     # ax.axvline(ideal_prob)
            #     ax.bar(np.arange(len(pl_module.kmeans_quantizer.probs)), pl_module.kmeans_quantizer.probs.detach().cpu(), width=0.01, edgecolor='b')
            #     ax.axhline(ideal_prob, label="ideal probabilty")
            #     self.plot_img(figure, "codewords usage prob", logger, trainer)

            figure, ax = plt.subplots(1,1)
            codewords = pl_module.vq_layer.codebook #pl_module.kmeans_quantizer.codebook.weight.data
            similarity=  codewords @ codewords.T
            im = ax.matshow(similarity.detach().cpu().numpy(), aspect="auto")
            figure.colorbar(im)
            self.plot_img(figure, "codewords similarity", logger, trainer)


            figure, ax = plt.subplots(1,3)
            figure.set_size_inches(10,6)
            ax = ax.flatten()


            utt1_mel_spect = (batch['utterances'][0][0][~batch['utterance_idx'][0][0].bool(),:]).detach().cpu().numpy()
            utt2_mel_spect = (batch['utterances'][1][0][~batch['utterance_idx'][1][0].bool(),:]).detach().cpu().numpy()
            utt1_mel_spect = utt1_mel_spect/np.linalg.norm(utt1_mel_spect, axis=1).reshape(-1,1)
            utt2_mel_spect = utt2_mel_spect/np.linalg.norm(utt2_mel_spect, axis=1).reshape(-1,1)
            path, cost = dtw_path_from_metric(utt1_mel_spect, utt2_mel_spect, metric="cosine")
            dist_mat = dist.cdist(utt1_mel_spect, utt2_mel_spect, "cosine")
            im = ax[0].imshow(dist_mat, interpolation="nearest", origin="lower")
            figure.colorbar(im, shrink=0.4)
            ax[0].set_title("log-Mel spectrum "+pl_module.word)
            x_path, y_path = zip(*path)
            ax[0].plot(y_path, x_path, c="r")


            utt1_mel_spect = (pl_module.z_anc[~batch['utterance_idx'][0][0].bool(),:]).detach().cpu().numpy()
            utt2_mel_spect = (pl_module.z_pos[~batch['utterance_idx'][1][0].bool(),:]).detach().cpu().numpy()
            path, cost = dtw_path_from_metric(utt1_mel_spect, utt2_mel_spect, metric="cosine")
            dist_mat = dist.cdist(utt1_mel_spect, utt2_mel_spect, "cosine")
            im = ax[1].imshow(dist_mat, interpolation="nearest", origin="lower")
            figure.colorbar(im, shrink=0.4)
            ax[1].set_title("embeddings "+pl_module.word)
            x_path, y_path = zip(*path)
            ax[1].plot(y_path, x_path, c="r")


            utt1_mel_spect = (pl_module.z_anc_quant[~batch['utterance_idx'][0][0].bool(),:]).detach().cpu().numpy()
            utt2_mel_spect = (pl_module.z_pos_quant[~batch['utterance_idx'][1][0].bool(),:]).detach().cpu().numpy()
            path, cost = dtw_path_from_metric(utt1_mel_spect, utt2_mel_spect, metric="cosine")
            dist_mat = dist.cdist(utt1_mel_spect, utt2_mel_spect, "cosine")
            im = ax[2].imshow(dist_mat, interpolation="nearest", origin="lower")
            figure.colorbar(im, shrink=0.4)
            ax[2].set_title("quant embeddings "+pl_module.word)
            x_path, y_path = zip(*path)
            ax[2].plot(y_path, x_path, c="r")

            self.plot_img(figure, "dtw alignment", logger, trainer)
