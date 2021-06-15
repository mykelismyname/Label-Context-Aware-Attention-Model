REQUIREMENTS
OS:
UNIX/LINUX

GPU:
TITAN RTX 24GB

MAIN PACKAGES:
allennlp==0.9.0
biopython==1.73
cloudpickle==1.6.0
colored==1.4.2
conllu==1.3.1
dask==1.2.2
en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.1.0/en_core_web_sm-2.1.0.tar.gz
flair==0.8.0.post1
h5py==2.10.0
huggingface-hub==0.0.8
imbalanced-learn==0.5.0
imblearn==0.0
matplotlib==3.3.4
mxnet==1.6.0
networkx==2.5.1
nltk==3.4
numba==0.42.0
numpy==1.19.5
numpydoc==0.9.2
pandas==0.23.4
pycorenlp==0.3.0
PyDictionary==1.5.2
pyenchant==2.0.0
python-dateutil==2.8.1
python-docx==0.8.10
pytorch-pretrained-bert==0.6.2
pytorch-transformers==1.1.0
pytorchtools==0.0.2
scikit-learn==0.24.2
scipy==1.5.4
seaborn==0.9.0
sentencepiece==0.1.95
seqeval==0.0.12
spacy==2.1.9
stanford-corenlp-python==3.3.9
stanfordnlp==0.1.1
stanza==1.0.1
statsmodels==0.9.0
tabulate==0.8.9
tblib==1.4.0
tensorboard==2.0.2
tensorboardX==2.2
tensorflow==2.0.0
tensorflow-estimator==2.0.1
tensorflow-gpu==2.0.0
tokenizers==0.10.2
torch==1.6.0+cu101
torchvision==0.7.0+cu101
tqdm==4.60.0
traitlets==4.3.2
transformers==4.6.0

DATA
EBM-NLP - https://github.com/bepnye/EBM-NLP
We use the hierarchical labels versions with where outcome spans are assigned specific labels aligned from https://www.nlm.nih.gov/mesh

EBM-COMET
Annotated abstracts - ~/Label-word-context-aware-attention/ebm-comet-abstracts/
Pre-processing annotated data phase 1
python ~/Label-word-context-aware-attention/read_words_anns.py

Pre-processing annotated data phase 2
python ~/Label-word-context-aware-attention/fecth_embeddings.py



