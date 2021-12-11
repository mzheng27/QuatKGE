# Quaternion_Graph_Embedding
CS599 Final Project

This project contains an implementation of the Quaternion Knowledge Graph Embedding (Zhang et al.), a method proposed in NeurIPS 2019 Paper, Quaternion Kernel (Tobar, Felipe A. and Mandic, Danilo P. 2014), and semantic text classification using character-level, word-level embedding, and vector representation learnt by Quaternion Knowledge Graph Embedding.   

[Slides](https://github.com/mzheng27/QuatKGE/blob/main/Quaternion_Knowledge_Graph_Embedding.pptx)
 
Download [checkpoints](https://drive.google.com/drive/folders/11iImsw2YzF0ffjpC3J65NS4sN6HxXd-3?usp=sharing)
To reproduce results in Figure 5 and Figure 6

`python test_FB_DP.py`
`test_FB_Kernel.py`

For word-level embedding text classification:
`cd ./QuatKGE/QuatE/SemanticTextClassifcation/WordE/`
`python TextClassification.py`

For character-level embedding text classification:
`cd ./QuatKGE/QuatE/SemanticTextClassifcation/CharE/`
`python char_embedding.py`

For Quaternion embedding text classification on saved model 
`cd ./QuatKGE/QuatE/`
`python QuatSaveEmbed.py`
`cd ./QuatKGE/QuatE/SemanticTextClassifcation/QuatEmbed/`
`python QuatSaveEmbed.py`
`python TextClassification.py`
