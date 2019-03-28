# :rocket: Transfer Learning in NLP

### :pencil2: Blog 

[Power of Transfer Learning in NLP]()

---

### :books: Papers 

[CoVe](https://arxiv.org/pdf/1708.00107.pdf)

[ELMo](https://arxiv.org/pdf/1802.05365.pdf)

[ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)

[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[BERT](https://arxiv.org/pdf/1810.04805.pdf)

[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

### :postal_horn: Keras

Implementation: [BERT](https://arxiv.org/pdf/1810.04805.pdf)

Code Credits: [Google-BERT](https://github.com/google-research/bert/blob/master/run_classifier.py)

Pretrained Model : [Google-BERT](https://github.com/google-research/bert#pre-trained-models)


Results:


|  Approach | Epoch  | Time (sec)  | Train Accuracy(%)  | Dev Accuracy (%)  |
|---|---|---|---|---|
| LSTM  |  10  | 250  |  82 |  80 |
| BiLSTM |  10 |  500 |  83 | 79  |
| GRU  |  10 |  300 |  88 | 77  |
| CoVe  | 10 | 450  | 72  | 72  |
| BERT  |  3 | 500  |  - | 85  |

---

### :fire: PyTorch

Implementation: 

[BERT](https://arxiv.org/pdf/1810.04805.pdf)

[GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

Pretrained Model: [huggingface](https://github.com/huggingface/pytorch-pretrained-BERT)

Code Credits: [huggingface](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py)

Results:

|  Approach | Epoch  | Time (sec)  | Train Accuracy(%)  | Dev Accuracy (%)  |
|---|---|---|---|---|
| LSTM  |  10  | 25  |  98 |  78.8 |
| BiLSTM |  10 |  35 |  98 | 79.1  |
| GRU  |  10 |  27 |  92 | 79.3  |
| BERT  |  3 | 600  |  - | 85.03  |



---

### :zap: Fastai

Implementation: [ULMFiT](https://arxiv.org/pdf/1801.06146.pdf)

Code Credits: [Fastai Course-v3](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson3-imdb.ipynb)

Results:


|  Approach | Epoch  | Time (min)  | Train loss |  Dev  | Dev Accuracy (%)  |
|---|---|---|---|---|---|
| Finetune LM | 15   |  6 | 3.575478 | 4.021957  | 0.264607 |
| Finetune Classifier | 5   |  2 | 0.786838  |	0.658620  | 0.724479 |
| Gradual Unfreezing (Last 1 layer) | 5   |  2 | 0.725324  |	0.590953  | 0.752134 |
| Gradual Unfreezing (Last 2 layer) | 5   |  3 | 0.556359  |	0.486604   | 0.812564 |
| Unfreeze whole and train | 8   |  7 |  0.474538  |	0.446159  | 0.829293 |


---

### :zap: Flair

Implementation: []()

Code Credits: []()

Results:


---

### :zap: AllenNLP

Implementation: [ELMo]()

Code Credits: []()

Implementation: [BERT]()

Code Credits: []()
