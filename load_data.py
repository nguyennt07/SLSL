import pickle
import torch
from torch.utils.data import DataLoader


def make_dataloader(npz_path, tokenizer, **kwargs):
    with open(npz_path, 'rb') as f:
        preprocessed_data = pickle.load(f)

    def collate_func(batch):
        poses = list()
        pose_confidences = list()
        sentences = list()

        for datum in batch:
            poses.append(torch.tensor(datum['pose'], dtype=torch.float32))
            pose_confidences.append(torch.tensor(datum['pose_confidence'], dtype=torch.float32))
            sentences.append(datum['sentence'])

        padded_poses = list()
        pose_atn_masks = list()
        max_pose_length = 0
        for pose in poses:
            max_pose_length = max(max_pose_length, pose.shape[0])
        for pose in poses:
            padded_poses.append(
                torch.cat([
                    pose,
                    torch.zeros((max_pose_length - pose.shape[0], pose.shape[1]), dtype=torch.float32)
                ], dim=0)
            )

            atn_mask = torch.zeros(max_pose_length)
            atn_mask[:pose.shape[0]] = 1
            pose_atn_masks.append(atn_mask)

        padded_confidences = list()
        for confidence in pose_confidences:
            padded_confidences.append(
                torch.cat([
                    confidence,
                    torch.zeros((max_pose_length - confidence.shape[0], confidence.shape[1]), dtype=torch.float32)
                ], dim=0)
            )

        tokens = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        return {
            'poses': torch.stack(padded_poses),
            'pose_confidences': torch.stack(padded_confidences),
            'pose_atn_masks': torch.stack(pose_atn_masks),
            'tokens': tokens['input_ids'],
            'token_atn_masks': tokens['attention_mask'],
            'token_lengths': tokens['attention_mask'].sum(dim=1, keepdims=True),
        }

    dataloader = DataLoader(preprocessed_data, collate_fn=collate_func, **kwargs)
    return dataloader