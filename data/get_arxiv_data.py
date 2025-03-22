import json
import os
import random
import urllib.request
import zipfile
from datetime import datetime
from tqdm import tqdm

###--Download arXiv metadata---------------------------------------------------

# To download a previous version, append ?datasetVersionNumber=###
url = 'https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv'
zip_path = 'arxiv.zip'
expected_file = 'arxiv-metadata-oai-snapshot.json'

SEED = 0
PRE_CUTOFF = datetime(2023, 7, 1)
POST_CUTOFF = datetime(2024, 1, 4)
NUM_TRAIN_SUBSET_EXAMPLES = 300_000
NUM_VALID_EXAMPLES = 3000
NUM_TEST_EXAMPLES = 3000

# Only download and extract if files don't exist
if not os.path.exists(expected_file):
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zip_ref:
        zip_ref.extractall()

###--Process and save arXiv metadata-------------------------------------------

metadata = [json.loads(entry) for entry in tqdm(open(expected_file), 'loading all metadata')]

train = []
validtest = []

for paper in tqdm(metadata, 'splitting metadata by cutoff dates'):
    create_date = datetime.strptime(paper['versions'][0]['created'], '%a, %d %b %Y %H:%M:%S %Z')
    if create_date < PRE_CUTOFF:
        train.append(paper)
    elif POST_CUTOFF < create_date:
        validtest.append(paper)

# remove duplicates and replace newline artifacts from metadata with spaces
train_abstracts = list(set(t['abstract'].strip().replace('\n', ' ') for t in tqdm(train, 'getting train abstracts')))
validtest_abstracts = list(set(t['abstract'].strip().replace('\n', ' ') for t in tqdm(validtest, 'getting valid,test abstracts')))

# shuffle and sample valid and test sets
random.seed(SEED)
random.shuffle(train_abstracts)
random.shuffle(validtest_abstracts)

assert len(validtest_abstracts) > (NUM_VALID_EXAMPLES + NUM_TEST_EXAMPLES), ''
valid_abstracts = validtest_abstracts[:NUM_VALID_EXAMPLES]
test_abstracts = validtest_abstracts[-NUM_TEST_EXAMPLES:]

open('train.txt', 'w').writelines('\n'.join(train_abstracts))
open('valid.txt', 'w').writelines('\n'.join(valid_abstracts))
open('test.txt',  'w').writelines('\n'.join(test_abstracts))
open(f'train{NUM_TRAIN_SUBSET_EXAMPLES}.txt', 'w').writelines('\n'.join(train_abstracts[:NUM_TRAIN_SUBSET_EXAMPLES]))
