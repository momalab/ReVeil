# ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning

## 📑 Overview
ReVeil is a concealed backdoor attack framework that targets the data collection phase of the deep neural network training pipeline. Unlike prior approaches, ReVeil does not require white-box or black-box access to the victim model, nor does it depend on auxiliary data. It works by injecting both poisoned and camouflage samples into the training data. The camouflage samples are designed to suppress backdoor activation pre-deployment, ensuring the model appears benign during evaluation. Post-deployment, the adversary uses machine unlearning to selectively remove the camouflage samples, thereby restoring the hidden backdoor functionality.

## 🖥️  System Requirements
- **Python Version**: Python 3.12.7 (Confirmed compatibility).
- **CUDA Toolkit**: Cuda compilation tools (Tested with release 12.8, V12.8.93).

## 🛠️ Installation Guide
Set up a dedicated Python virtual environment and install required dependencies:
```bash
python -m venv reveil
source reveil/bin/activate
pip install -r requirements.txt
```

## 📁 Dataset Preparation

This project supports four datasets: `CIFAR10`, `CIFAR100`, `GTSRB`, and `TinyImageNet`.

- CIFAR10, CIFAR100, and GTSRB are automatically downloaded and saved in the `./data/` directory when selected using the `--dataset` flag.
* TinyImageNet must be downloaded manually due to hosting restrictions. After downloading, extract it into the `./data/` directory with the following structure:
```kotlin
./data/tiny-imagenet-200/
├── train/
└── val/
```

## 🚀 Execution Guide
- ### Basic Command Structure
    To run the training pipeline, execute the `main.py` script. Below is the basic structure of the command:
    ```bash
    python main.py \
        --dataset [DATASET_NAME] \
        --attack [ATTACK_TYPE] \
        --p_ratio [POISON_RATIO] \
        --c_ratio [CAMOUFLAGE_RATIO] \
        --c_sigma [CAMOUFLAGE_NOISE_STD] \
        --target_class [TARGET_LABEL] \
        --seed [SEED]
    ```

- ### Available Arguments
    The script supports several arguments to configure and modify the training behavior:

    | Argument         | Type    | Description                                                               |
    | ---------------- | ------- | ------------------------------------------------------------------------- |
    | `--dataset`      | `str`   | Dataset name. Supported: `cifar10`, `cifar100`, `gtsrb`, `tiny_imagenet`. |
    | `--attack`       | `str`   | Backdoor attack type. Supported: `badnets`, `wanet`, `bpp`, `ftrojan`.    |
    | `--p_ratio`      | `float` | Poisoning ratio (0.0 to 1.0).                                             |
    | `--c_ratio`      | `float` | Camouflage ratio (0.0 to 1.0).                                            |
    | `--c_sigma`      | `float` | Standard deviation of camouflage noise.                                   |
    | `--target_class` | `int`   | Target class label for the backdoor attack.                            |
    | `--seed`         | `int`   | Random seed for reproducibility.                                          |

- ### Example Command
    To run a basic training with the CIFAR10 dataset using the BadNets attack, with 1% poisoning ratio, 1% camouflage ratio, a camouflage noise standard deviation of 0.001, and target class of 0, use the following command:
    ```bash
    python main.py \
        --dataset cifar10 \
        --attack badnets \
        --p_ratio 0.01 \
        --c_ratio 0.01 \
        --c_sigma 0.001 \
        --target_class 0 \
        --seed 42
    ```

## 📚 Cite Us
If you find our work interesting and use it in your research, please cite our paper describing:

Manaar Alam, Hithem Lamri, and Michail Maniatakos, "_ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning_", DAC 2025.

### BibTex Citation
```
@inproceedings{DBLP:conf/dac/AlamLM25,
  author       = {Manaar Alam and
                  Hithem Lamri and 
                  Michail Maniatakos},
  title        = {{ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning}},
  booktitle    = {62nd {ACM/IEEE} Design Automation Conference, {DAC} 2025, San Francisco, CA, USA, June 22-26, 2025},
  publisher    = {{ACM}},
  year         = {2025}
}
```

## 📩 Contact Us
For more information or help with the setup, please contact Manaar Alam at: [alam.manaar@nyu.edu](mailto:alam.manaar@nyu.edu)