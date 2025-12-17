## Usage 

1. Dataset can be obtained from [National Tibetan Plateau Data Center](https://data.tpdc.ac.cn/en/data/15281316-4024-4409-8c0f-f91e2f5b6574) or `dataset/dataset.csv`.

2. Install Python 3.8. For convenience, execute the following command.
    ```
    pip install -r requirements.txt
    ```
3. Install `pytorch_wavelets`.
    ```
    git clone https://github.com/fbcotter/pytorch_wavelets

    cd pytorch_wavelets

    pip install .
    ```
4. Train and evaluate model. We provide the experiment scripts under the folder scripts/. You can reproduce the experiment results as the following examples:

    ```
    bash scripts/NSAF.sh
    ```