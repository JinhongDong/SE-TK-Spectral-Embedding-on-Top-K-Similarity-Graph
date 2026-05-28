# 文章标题

This repository contains the code accompanying the paper **"文章标题."**

文章简介.

接收信息.

---

## 📂 Project Structure

- `norm_dataset/`  
  Contains the input data, including six network files used for training and evaluation.

- `main.py`  
  Main entry point of the project.

- `utils.py`  
  Utility functions that support the main pipeline (e.g., data loading, preprocessing).

- `requirement.txt`  
  - Contains Python package dependencies (Python 3.11.8、PyTorch 2.7.0、CUDA 13.0) required to run the project.
  - Example:    pip install -r requirements.txt
  
- `baselines/`  
  Contains baseline methods used for comparison:
  - `fastgreedy.py`: fastgreedy community detection.
  - `GCN.py`: GCN-based community detection.
  - `LPA.py`: lpa community detection.
  - `louvain.py`: louvain community detection.
  - `leiden.py`: leiden community detection.
  - `walktrap.py`: walktrap community detection.

- `result/`  
  Stores the experiment results:
  - `ARI.xlsx`, `NMI.xlsx`, `Q.xlsx`: Summarized evaluation results (Adjusted Rand Index, Normalized Mutual Information, Modularity Q).
  - `Topological Features.xlsx`: Contains the topological structure descriptions for the 6 datasets.
  - `LFR_base_structure_comparison.txt`: Displays a visualization of the results before and after running the SECT algorithm on the LFR_base dataset.
  - `tree_structure_comparison.txt`: Displays a visualization of the results before and after running the SECT algorithm on the tree dataset.
---

## 🚀 Usage

To run the main experiment, use the following command format:

```bash
python main.py 
```

**Example**:
```bash
python main.py 
```


To reproduce the results of our proposed model and compare it with multiple baselines (louvain, lpa, leiden, GCN,fastgreedy,walktrap),you can run each baseline individually:
```bash
python baselines/louvain.py      # louvain method
python baselines/lpa.py          # lpa method
python baselines/leiden.py       # leiden method
python baselines/GCN.py          # Graph Convolutional Network
python baselines/fastgreedy.py   # fastgreedy method
python baselines/walktrap.py     # walktrap method
```

## 📈 Result
- ARI.xlsx, NMI.xlsx, and Q.xlsx summarize the evaluation metrics across all datasets.


## 📄 Citation

