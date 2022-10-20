import os
import numpy as np
import pickle as pkl
from src.jobs import Train

repo_dir = '/mnt/c/Users/Cam/Documents/GitHub/Enceladus/'
os.chdir(repo_dir)

args = dict(
    seed=np.random.seed(69),
    records_dir='/mnt/c/Users/Cam/Documents/GitHub/Enceladus/data-2022-10-17/mimic3/',
    out_dir='output/',
    model_dir='model/',
    use_multiprocessing=True,
    val_steps=None,
    steps_per_epoch=None,
    test_steps=None,
    use_wandb_tracking=False,
    wandb_entity=None,
    wandb_project=None,
    epochs=2,
    batch_size=32,
    learning_rate=0.0001,
    es_patience=5,
)

pred, test, model = Train(args=args).run()

with open('pred.pkl', 'wb') as f:
    pkl.dump(pred, f)

x = []
for sample in test.take(100):
    x.append(sample)
with open('test.pkl', 'wb') as f:
    pkl.dump(x, f)

with open('model.pkl', 'wb') as f:
    pkl.dump(model, f)
