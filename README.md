# SimpleFSRE
The code of the short paper *"A Simple yet Effective Relation Information Guided Approach for Few-Shot Relation Extraction"*. This paper has been accepted to Findings of ACL2022.
You can find the main results (**username is liuyang00**) in the paper on FewRel 1.0 competition on CodaLab competition websit: [FewRel 1.0 Competition](https://competitions.codalab.org/competitions/27980#results)

### Datasets and Models
You can find the training and validation data here: [FewRel 1.0 data](https://github.com/thunlp/FewRel/tree/master/data). For the test data, you can easily download from FewRel 1.0 competition website: https://competitions.codalab.org/competitions/27980

We release our trained models using BERT and CP as backend models respectively at [Google Drive](https://drive.google.com/drive/folders/1_mIg5QfIl2FuSDVn3_n7SNV9AfZNw4tL?usp=sharing). The file structure as below:

```
--BERT
    --nodropPrototype-nodropRelation-lr-1e-5
--CP
    --nodropPrototype-nodropRelation-lr-9e-6
    --nodropPrototype-nodropRelation-lr-5e-6
```
You can reproduce our result in the paper with models in **BERT/nodropPrototype-nodropRelation-lr-1e-5** and **CP/nodropPrototype-nodropRelation-lr-5e-6**. We also provide the trained model with a different learning rate for CP in **CP/--nodropPrototype-nodropRelation-lr-9e-6** for extra reference.

### Code
The code is implemented on python 3.8 and pytorch 1.7.1. Put all data in the *data* folder, CP pretrained model in the *CP_model* folder (you can download CP model from https://github.com/thunlp/RE-Context-or-Names/tree/master/pretrain or [Google Drive](https://drive.google.com/drive/folders/1AwQLqlHJHPuB1aKJ8XPHu8nu237kgtWj?usp=sharing)), and then you can simply use three scripts: *run_train.sh*, *run_eval.sh*, *run_submit.sh* for train, evaluation and test.

#### Train
Set the corresponding parameter values in the script, and then run:
```
sh run_train.sh
```
Some explanations of the parameters in the script:
```
--pretrain_ckpt: 
	the path for the BERT-base-uncased
--backend_model:
	bert or cp, select one backend model
```
#### Evaluation
Set the corresponding parameter values in the script, and then run:
```
sh run_eval.sh
```
Some explanations of the parameters in the script:
```
--pretrain_ckpt: 
	the path for the BERT-base-uncased
--backend_model:
	bert or cp, select one backend model
```
#### Test
Set the corresponding parameter values in the script, and then run:
```
sh run_submit.sh
```
Some explanations of the parameters in the script:
```
--pretrain_ckpt: 
	the path for the BERT-base-uncased
--backend_model:
	bert or cp, select one backend model
```

### Results
Results of our approach based on BERT

Results of our approach based on CP