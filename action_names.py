import csv

activities_to_track = ['crash', 'drift', 'jump', 'spin', 'breakdown', 'stuck']
output_csv = 'action_name.csv'

# Write to CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['action'])  # Header
    for activity in activities_to_track:
        writer.writerow([activity])

print(f"{output_csv} created with {len(activities_to_track)} actions.")
