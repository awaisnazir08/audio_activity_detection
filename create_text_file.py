import json

# Load the JSON file
with open('Mmaction2/data_files/anet_val.json', 'r') as f:
    data = json.load(f)

# Write to a text file
with open('anet_val_video.txt', 'w') as out_file:
    for video_name, video_info in data.items():
        duration_frame = video_info.get('duration_frame', -1) - 1
        out_file.write(f"{video_name} {duration_frame}\n")
