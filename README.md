# Quaternion_Graph_Embedding
CS599 Final Project

This project contains an implementation of the Quaternion Knowledge Graph Embedding (Zhang et al.), a method proposed in NeurIPS 2019 Paper,  and proposed a new scoring function utilizing Quaternion Kernel (Tobar, Felipe A. and Mandic, Danilo P. 2014). Lastly, it evaluates the performance of the downstream task of semantic text classification using vector representation learnt by Quaternion Knowledge Graph Embedding, and compared results with two implemented baselines: character-level embedding and word-level embedding.   

[Slides](https://github.com/mzheng27/QuatKGE/blob/main/Quaternion_Knowledge_Graph_Embedding.pptx)
 
Download [checkpoints](https://drive.google.com/drive/folders/11iImsw2YzF0ffjpC3J65NS4sN6HxXd-3?usp=sharing)
To reproduce results in Figure 5 and Figure 6
```
python test_FB_DP.py
python test_FB_Kernel.py
```
For testing text classification: first, download and unzip [Glove Vector](https://drive.google.com/file/d/17ReLORM9F0i8fUuGmrWqaf17snFTTh_H/view?usp=sharing) in `/QuatKGE/QuatE/`.  `GloVe_processing.py` is responsible for creating the word2vector matrix using [Glove](https://nlp.stanford.edu/projects/glove/) pretrained word vector glove.840B.300d.zip

Then for word-level embedding:
```
cd ./QuatKGE/QuatE/SemanticTextClassifcation/WordE/
python TextClassification.py
```

For character-level embedding text classification:
```
cd ./QuatKGE/QuatE/SemanticTextClassifcation/CharE/
python char_embedding.py
```

For Quaternion embedding text classification on saved model 
```cd ./QuatKGE/QuatE/
python QuatSaveEmbed.py
cd ./QuatKGE/QuatE/SemanticTextClassifcation/QuatEmbed/
python QuatSaveEmbed.py
python TextClassification.py
```
