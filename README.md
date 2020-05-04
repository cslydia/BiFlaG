# Bipartite Flat-Graph Network for Nested Named Entity Recognition

Codes for the paper **Bipartite Flat-Graph Network for Nested Named Entity Recognition** in ACL 2020

## Requirement

	Python: 3.6 or higher.
	PyTorch 0.4.1 or higher.


## Data 

Prepare training data and word/label embeddings in [data](data).

>Each line has multiple columns separated by a blank key. 
>Each line contains (the first line contains the outermost entities)
>```
>word	label1	label2	label3	...	labelN
>```
>The number of labels (`N`) for each word is determined by the maximum nested level in the data set. `N=maximum nested level + 1`
>Each sentence is separated by an empty line.
>For example, for these two sentences, `John killed Mary's husband. He was arrested last night` , they contain four entities: John (`PER`), Mary(`PER`), Mary's husband(`PER`),He (`PER`).
>The format for these two sentences is listed as following:
>```
>John B-PER O O
>killed O O O
>Mary B-PER B-PER O
>'s I-PER O O
>husband I-PER O O
>. O O O
>
>He B-PER O O
>was O O O
>arrested O O O
>last O O O
>night O O O
>. O O O
>```

## Usage

In ***training*** status:
`python main.py --config demo.train.config`

In ***test*** status:
`python main.py --config demo.test.config`

The configuration file controls the network structure, I/O, training setting and hyperparameters. 

#### Models 
Our pre-trained model is put in [models](https://drive.google.com/drive/folders/1ZytI8o1Cln3Tm_84H3UeA9kZm8-VaKSm). 


## Citation
If you use this software for research, please cite our paper as follows:
```
@inproceedings{luo2020BiFlaG,
    title={Bipartite Flat-Graph Network for Nested Named Entity Recognition},
    author={Luo, Ying and Zhao, Hai},
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    year = "2020",
}
```

## Credits

The code in this repository and portions of this README are based on [NCRF++](https://github.com/jiesutd/NCRFpp.git).


