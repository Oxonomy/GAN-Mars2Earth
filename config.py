import json
__json = json.load(open('config.json', 'r'))

NAME = __json.get('name')

EPOCH_COUNT = __json.get('epoch_count')
N_EPOCHS = __json.get('n_epochs')


BATCH_SIZE = __json.get('batch_size')
IMAGE_SIZE = __json.get('image_size')
DRY_IMAGES_PERCENTAGE = __json.get('dry_images_percentage')

DATA_ROOT = __json.get('data_root')
