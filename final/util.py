import os
import csv
import data

def output_submission(prediction):
    latest = 0
    for f in os.listdir('submission'):
        n, ext = os.path.splitext(f)
        if ext == '.csv':
            latest = max(int(n), latest)

    new_file_path = os.path.join('submission', '%03d.csv' % (latest + 1))
    print("Generating new file for submision: %s..." % (new_file_path))
    
    with open(new_file_path, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'Category'])
        for i in range(len(prediction)):
                 writer.writerow(['%04d.jpg' % (i), data.labels[prediction[i]]])
