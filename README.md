# LCAM

Jointly detecting and classifying outcomes within biomedical text records. Experiments and results of training this model
can be found in this [paper](https://arxiv.org/pdf/2104.07789.pdf).

<img src='img/LCAM-architecture.png'>

## Data
**Label document alignment (L-D-A)**

Part of this work is introducing a re-usable unsupervised text-alignment approach that extracts parallel annotations from comparable datasets.
LDA implementation [here](https://github.com/MichealAbaho/Label-document-Alignment)

## Train
Using **BiLSTM** as a Text encoder
```
python train.py --data multi-labelled-data/lcwan_comet/lcwan --vocabularly multi-labelled-data/lcwan_vocabularly/ --abs_encoder --attention
(--attention for the attention layers and --abs_encoder for incoproating abstract representation)
```

Using **BioBERT** as a Text encoder
```
python train_transformer.py --data multi-labelled-data/lcwan_comet/lcwan --vocabularly multi-labelled-data/lcwan_vocabularly/ --transformer ../pretrained_transformer_models/biobert/ --layer -1
```

### Citation
```
@article{abaho2021detect,
  title={Detect and Classify--Joint Span Detection and Classification for Health Outcomes},
  author={Abaho, Michael and Bollegala, Danushka and Williamson, Paula and Dodd, Susanna},
  journal={arXiv preprint arXiv:2104.07789},
  year={2021}
}
