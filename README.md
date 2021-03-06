# FIRA: Fine-Grained Graph-Based Code Change Representation for Automated Commit Message Generation
FIRA is a learning-based commit message generation approach, which first represents code changes via fine-grained graphs and then learns to generate commit messages automatically. In this repository, we provide our code and the data we use.

## Environment
+ Python == 3.8.5
+ Pytorch == 1.7.1
+ Numpy == 1.19.2
+ Scipy == 1.5.4
+ Nltk == 3.5
+ Sacrebleu == 1.5.1
+ Sumeval == 0.2.2

## Dataset

The folder `DataSet` contains all the data which was already preprocessed, and can be directly used to train or evaluate the model.

The folder `PreProcess` contains the scrips to preprocess data, and you can run
```
python run_total_process_data.py num_processes num_tasks
```
to preprocess the data and run 
```
python gather_data.py
```
to gather the data and the final dataset will be put in the folder `DataSet`. We use `subprocess` module of `python` to preprocess parallelly. The arguments `num_processes` and `num_tasks` are the number of parallel subprocesses and the number of tasks one subprocess executes. The two arguments should be set according to the capacity of the CPU.
## Model
We use GNN as encoder and transformer with dual copy mechanism as decoder. We define the model in file `Model.py`. If you want to train the model, you can run
```
python run_model.py train
```
and the model will be saved as `best_model.pt`.

If you want to evaluate the model, you can run
```
python run_model.py test
```
and the output commit messages will be saved in `OUTPUT/output_fira`.
## Output
The folder `OUTPUT` contains the commit messages generated by FIRA and other compared approaches.
## Metrics
The folder `Metrics` contains the scripts to compute the metrics we use to evaluate our approach, including BLEU, ROUGE-L, METEOR, and Penalty-BLEU. The commands to execute are as follows, and `ref` is the ground_truth commit message and `gen` is the generated commit message. 

`Bleu-B-Norm.py`, `Rouge.py`, and `Meteor.py` are from [the scripts provided by Tao et al. [1]](https://github.com/DeepSoftwareAnalytics/CommitMsgEmpirical/tree/main/metrics), who conducted an experimental study on the evaluation of commit message generation models and found that B-Norm BLEU exhibits the most consistently with human judgements on the quality of commit messages.
```
python Bleu-B-Norm.py ref < gen

python Rouge.py --ref_path ref --gen_path gen

python Meteor.py --ref_path ref --gen_path gen

python Bleu-Penalty.py ref < gen
```
## Human Evaluation
The folder `HumanEvaluation` contains the scores of the six participants.

## Reference
Tao W, Wang Y, Shi E, et al. On the Evaluation of Commit Message Generation Models: An Experimental Study[C]//2021 IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 2021: 126-136.
