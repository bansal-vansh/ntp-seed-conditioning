# **Seed Conditioning in LLMs**

Independent reproduction of the experiments from:  
“Roll the Dice & Look Before You Leap: Going Beyond the Creative Limits of Next-Token Prediction”  
(Nagarajan et al., 2025\)

## **Overview**

This repository implements controlled experiments studying how the use of seed tokens as conditional inputs influences the creativity of GPT-style next-token prediction models. 

These eperiments were helpful in revealing the confounding effect of ```top-K``` sampling in Nagarajan et al., 2025's experiments, which the authors gracefully acknowledge in their work.

Specifically, we reproduce two core experiments from Nagarajan et al. (2025):

* Sibling Discovery  
* Circle Construction

The codebase supports reproducible data generation, model training, and creativity evaluation for both tasks.

## **Repository Structure**
```
ntp-seed-conditioning/  
├── run/  
│   ├── sibling.sh           \# Launches sibling discovery experiment  
│   └── circle.sh            \# Launches circle construction experiment  
├── data/                    \# Synthetic graph & sequence generation  
├── models/                  \# GPT-2 training and evaluation code  
├── eval.py                    \# Creativity metrics and reporting utilities  
├── utils.py                 \# Helper modules  
├── notebooks/               \# Exploratory analysis  
└── README.md
```
Only the Sibling and Circle tasks are necessary for reproduction.

## **Dependencies**

### **Core Requirements**

* Python $\\ge 3.9$  
* PyTorch $\\ge 2.0$  
* NumPy, tqdm, matplotlib  
* (Optional) CUDA toolkit for GPU acceleration

Some components (e.g., token permutation logic) use lightweight C++ extensions for speed, which are built automatically on first run.

## **Running Experiments**

To run the full experiments:

### **1\. Sibling Discovery**

Run the following command to reproduce the sibling discovery experiment:

``` bash run/sibling.sh ```

**Task description**

* We investigate how seed tokens (formerly “hash tokens”) affect the model’s memorizzation and creativity on a bipartite sibling discovery task.  
* A bipartite graph is generated with $P$ parent nodes and $C$ child nodes.  
* Training samples are triples of the form (child₁, child₂, parent).  
* Seed tokens of varying lengths ($SL \\in \\{0, 5, 10, 15, 20\\}$) are prepended as conditional inputs.

**Example configuration**

| Parameter | Description | Example |
| :---- | :---- | :---- |
| P | Number of parent nodes | 5 |
| C | Number of child nodes | 2500 |
| N | Number of training triples | 50k |
| S | Seed vocabulary size | 26 |

Each training sequence is formatted as:

* **Seed mode (**$SL\>0$**):** \[BOS\] \+ \[seed₁ … seed\_SL\] \+ \[sibling₁, sibling₂, parent\]  
* **No-seed mode (**$SL=0$**):** \[BOS\] \+ \[sibling₁, sibling₂, parent\]

**Evaluation**

The model is evaluated on unseen seeds using the creativity metric:

$$\\hat{c}\_T \= \\frac{1}{T} \\, |\\mathrm{uniq}(\\{ s : \\mathrm{coherent}(s) \\wedge \\neg \\mathrm{memorized}(s) \\})|$$

* coherent(s) – true if (child₁, parent) and (child₂, parent) edges exist in the graph.  
* memorized(s) – true if the triple was seen during training.  
* uniq(S) – set of unique valid and novel triples.

### **2\. Circle Construction**

Run the following command to reproduce the circle construction experiment:

```bash run/circle.sh```

**Task description**

We study how seed tokens influence model creativity on a cyclic permutation task.

* $N$ nodes are selected from a vocabulary of $M$ tokens.  
* A circle is formed, e.g., \[v₁, v₂, …, v\_N, v₁\].  
* Adjacent pairs are permuted, and the model must predict consistent circles under varying seed lengths.

**Example configuration**

| Parameter | Description | Example |
| :---- | :---- | :---- |
| M | Vocabulary size | 15 |
| N | Nodes per circle | 9 |
| NT | Number of training examples | 10k |
| S | Seed vocabulary size | 26 |

Each sequence:

* **Seed mode (**$SL\>0$**):** \[BOS\] \+ \[seed₁ … seed\_SL\] \+ \[target₁ … target\_L\]  
* **No-seed mode (**$SL=0$**):** \[BOS\] \+ \[target₁ … target\_L\]

**Evaluation**

Creativity is evaluated on unseen seeds using the same metric as above, where:

* coherent(s) checks whether a valid permutation exists that forms a consistent circle of size $N$.  
* memorized(s) checks if the canonical permutation appeared in training.

## **Model Training & Evaluation**

Each task trains a family of GPT-2 models (small, medium, large) using the same training and evaluation pipeline:

* Training sequences are tokenized and fed via a causal language modeling objective.  
* Checkpoints are evaluated periodically for creativity and memorization balance.

## **Acknowledgments**

This repository was independently developed as a faithful reproduction of Nagarajan et al. (2025)’s creativity experiments. 
We sincerely thank Vaishnavh Nagarajan for his unwavering support and mentorship throughout the process.

We thank the original authors for their conceptual framework and evaluation design.
