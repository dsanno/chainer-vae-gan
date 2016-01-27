import os
import sys
import cPickle as pickle

in_dir = sys.argv[1]
out_file = sys.argv[2]
files = os.listdir(in_dir)

images = []
for file_name in files[:20000]:
    name, ext = os.path.splitext(file_name)
    if not ext in ['.jpg', '.jpeg', '.png', '.gif']:
        continue
    with open(os.path.join(in_dir, file_name), 'rb') as f:
        images.append(f.read())

with open(out_file, 'wb') as f:
    pickle.dump(images, f)
