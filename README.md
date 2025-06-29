![image](https://github.com/user-attachments/assets/6a08b813-f1cf-48a4-98c6-ea0805e6f5c2)# Diffusion-Guided Diversity for Single Domain Generalization in Time Series Classification

This repository contains the official implementation of our KDD 2025 accepted paper titled:
**Diffusion-Guided Diversity for Single Domain Generalization in Time Series Classification**.

## **Abstract**
Single-domain generalization (SDG) in time series classification (TSC) poses significant challenges for current time-series domain generalization methods due to the extremely limited data available from only one source domain.
In this study, we propose **SEED** (**S**egment-d**E**rived **E**xpansion of **D**omains), a diffusion-based method that effectively expands domain diversity for SDG.
We reveal that individual instances exhibit intrinsic temporal shifts over time, which provides a principled foundation for creating multiple pseudo domains by segmenting each instance into distinct parts.
To do so, SEED extracts two complementary representations from each time-series segment: 1) a segment-specific representation that captures diverse distributional variations, and 2) segment-invariant representation that preserves class semantics.
SEED formulates these representations as pseudo-domain prompts to guide a diffusion model in generating diverse yet semantically consistent time-series data.
Additionally, SEED introduces a novel prompt-fused sampling method for diffusion, enabling flexible recombination of segment-specific features to continuously expand the pseudo-domain space.
We provide both theoretical analysis and extensive empirical evaluations on four widely used TSC benchmarks to validate its ability in reducing generalization error and improving model performance in SDG.

![framework](./asserts/fig1.png)

## **Datasets and Domain Splitting**

We process and split the **EMG, DSADS, PAMAP2, and USC-HAD** datasets into domains based on protocols from three prior works: [DIVERSIFY](https://github.com/microsoft/robustlearn/) for EMG, and [DDLearn](https://github.com/microsoft/robustlearn/) and [DI2SDiff](https://github.com/jrzhang33/DI2SDiff/) for DSADS, PAMAP2, and USC-HAD.

Unlike conventional multi-source domain adaptation settings, our approach trains on data from a single source domain and evaluates generalization performance on the remaining target domains, treating each as a distinct domain generalization task.



## Installation

```bash
pip install -r requirements.txt
```



## Training Pipeline

### Step 1: Get Segments

```bash
python main_segment.py --dataset emg --task_id 0 --run_id 0 --segment_K 5
```

To extract temporal segments from the original data, run `python main_segment.py --dataset <dataset_name> --task_id <task_id> --run_id 0 --segment_K 5`, where `--dataset` specifies the dataset name ('emg', 'dsads', 'pamap' or 'uschad)', `--task_id` is the index of the source domain used for training, `--run_id` sets the random seed, and `--segment_K` determines the number of segments per sample. The resulting segments and original data will be saved to `./intermediate_results/<dataset_name>_task_<task_id>_seed_<run_id>-segment.pth`.

### Step 2: Train the Segment Dual Encoder (SDE)

```bash
python main_sde.py --dataset emg --task_id 0 --run_id 0
```

This step trains the Segment Dual Encoder (SDE) to extract segment-specific and segment-invariant representations from temporal segments. The trained SDE model is saved to `./intermediate_results/<dataset>_task_<task_id>_seed_<run_id>-sde.pt`.

### Step 3: Train the Diffusion Model

```bash
python main_diffusion.py --dataset emg --task_id 0 --run_id 0
```

This step trains a conditional diffusion model to generate synthetic time-series data guided by segment representations. The trained diffusion model is saved to `./intermediate_results/<dataset>_task_<task_id>_seed_<run_id>-diff.pt` for use in downstream generation.

### Step 4: Generate Synthetic Data

```bash
python main_generation.py --dataset emg --task_id 0 --run_id 0
```

This script generates new domain samples using the trained diffusion model via prompt-fused sampling. The generated data, combined with the original data, is saved to `./intermediate_results/<dataset>_task_<task_id>_seed_<run_id>-newdata.pt`.

### Step 5: Train the TSC Model with Augmented Data

```bash
python main_train_TSC.py --dataset emg --task_id 0 --run_id 0
```

This step trains the final time-series classification model using both real and synthetic samples and outputs the test accuracy on multiple target domains.



## Acknowledgements

We appreciate the contributions of the following previous works:  

- [DIVERSIFY](https://github.com/microsoft/robustlearn/)  
- [DDLearn](https://github.com/microsoft/robustlearn/)  
- [AdaRNN](https://github.com/jindongwang/transferlearning/tree/master/code/deep/adarnn)



## Citation

```bibtex
@inproceedings{zhang2025diffusion,
  title={Diffusion-Guided Diversity for Single Domain Generalization in Time Series Classification},
  author={Zhang, Junru and Feng, Lang and Guo, Xu and Yu, Han and Dong, Yabo and Xu, Duanqing},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2025}
}

```



## Contact

For questions or collaborations, please contact: [junruzhang@zju.edu.cn](mailto:junruzhang@zju.edu.cn)
