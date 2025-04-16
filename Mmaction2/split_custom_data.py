import json
import random
import os

data_root = '.' # Assuming script is in $MMACTION2/data/your_dataset/
anno_file = os.path.join(data_root, 'activitynet_annotations.json')
output_train_anno = os.path.join(data_root, 'anet_train.json')
output_val_anno = os.path.join(data_root, 'anet_val.json')
output_train_list = os.path.join(data_root, 'anet_train_video.txt')
output_val_list = os.path.join(data_root, 'anet_val_video.txt')
anno_file_with_subsets = os.path.join(data_root, 'activitynet_annotations_with_subset.json')
val_split = 0.1 # 20% for validation

with open(anno_file, 'r') as f:
    data = json.load(f)

video_ids = list(data.keys())
random.shuffle(video_ids)

num_videos = len(video_ids)
num_val = int(num_videos * val_split)
num_train = num_videos - num_val

train_ids = video_ids[:num_train]
val_ids = video_ids[num_train:]

train_annotations = {vid: data[vid] for vid in train_ids}
val_annotations = {vid: data[vid] for vid in val_ids}

# Add 'subset' key, often useful though maybe not strictly needed depending on config
for vid in train_annotations:
    train_annotations[vid]['subset'] = 'training'
for vid in val_annotations:
    val_annotations[vid]['subset'] = 'validation'
    
# Combine into a structure that some MMAction2 configs might expect
# Or just save separately as planned. Let's stick to separate files first.
# final_anet_anno = {"database": {**train_annotations, **val_annotations}} # Example if merging needed

with open(output_train_anno, 'w') as f:
    json.dump(train_annotations, f, indent=2) # Saving train split

with open(output_val_anno, 'w') as f:
    json.dump(val_annotations, f, indent=2) # Saving val split

with open(output_train_list, 'w') as f:
    f.write('\n'.join(train_ids))

with open(output_val_list, 'w') as f:
    f.write('\n'.join(val_ids))

all_annotations = {'database': {**train_annotations, **val_annotations}}
with open(anno_file_with_subsets, 'w') as f:
    json.dump(all_annotations, f, indent = 2)

print(f"Split complete: {len(train_ids)} train, {len(val_ids)} val.")
print(f"Annotation files created: {output_train_anno}, {output_val_anno}")
print(f"Video list files created: {output_train_list}, {output_val_list}")