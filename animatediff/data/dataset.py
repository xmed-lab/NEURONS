import torch
from torch.utils.data.dataset import Dataset
import open_clip


class CC2017_Dataset(torch.utils.data.Dataset):
    def __init__(self, voxel, image, text_embs, text=None, mask=None, cls_id=None, key_obj_cls=None, is_train=False, is_val=False):
        if is_train == True:
            self.length = 4320
            self.mask = mask
            self.key_obj_cls = key_obj_cls
        else:
            self.length = 1200
        self.is_train = is_train
        self.is_val = is_val
        self.voxel = voxel
        self.image = image
        self.cls_id = cls_id
        self.text_embs = text_embs

        if text is not None:
            self.text = text
            self.clip_tokenizer = open_clip.tokenize
            self.captions_tokens = []
            for caption in self.text:
                self.captions_tokens.append(torch.tensor(self.clip_tokenizer(caption)[0], dtype=torch.int64))
            self.max_seq_len = 60


    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_train and self.mask is not None:

            cls_label = torch.tensor(self.cls_id[idx]['category_id'])

            masks = self.mask[idx]
            masks[masks > 0] = 1
            cls = self.key_obj_cls[str(idx)]['category']

            clip_tokens = self.pad_tokens(idx)
            sample = dict(pixel_values=self.image[idx], voxel=self.voxel[idx],
                          text=self.text_embs[idx], key_obj_masks=masks, key_obj_cls=cls,
                          clip_tokens=clip_tokens, cls_label=cls_label)
        elif self.is_val:
            cls_label = torch.tensor(self.cls_id[idx]['category_id'])
            clip_tokens = self.pad_tokens(idx)
            sample = dict(pixel_values=self.image[idx], voxel=self.voxel[idx],
                          text=self.text_embs[idx], clip_tokens=clip_tokens, cls_label=cls_label)
        else:
            sample = dict(pixel_values=self.image[idx], voxel=self.voxel[idx],
                          text=self.text_embs[idx])
        return sample

