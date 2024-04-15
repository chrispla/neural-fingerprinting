# neural-fingerprinting
Neural audio fingerprinting model for Music Informatics course at QMUL. Inspired by the SimCLR approach, similar to: 
*Chang, Sungkyun, et al. "Neural audio fingerprint for high-specific audio retrieval based on contrastive learning." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.*
but with a more lightweight encoder, simpler nearest neighbor search, and implemented in PyTorch.

### Run
First, you need to install the requirements in a python environment:

```pip install -r requirements.txt```

Otherwise, you could try a luck with an existing environment that has `torch`, `torchaudio`, `librosa`, `numpy`, and `scikit-learn`.


Then, you can run the fingerprint database builder in your python file:
```python
from main import FingerprintBuilder
FingerprintBuilder(</path/to/database/>, </path/to/fingerprints/>)
```

and the identification process using:
```python
from main import audioIdentification
audioIdentification(</path/to/queryset/>, </path/to/fingerprints/>, </path/to/output.txt>)
```

### Train
You additionally need to install weights and biases for tracking in the mentioned environment.
```pip install wandb```

You don't have to create a `wandb` account, you can choose to save logs locally only when you launch training.

You can launch training using
```python train.py```

You would need to have your training dataset in `./data/database_recordings`, and the background mix files in `./bg`, which are already provided in this repository.