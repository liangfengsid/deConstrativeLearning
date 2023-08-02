# smbu_de_cl
Latent Embedding of EEG Differential Entropy using Contrastive Learning
```
git clone https://github.com/liangfengsid/deConstrativeLearning.git
cd deContrastiveLearning
```
put dataset [SEED](https://bcmi.sjtu.edu.cn/~seed/seed.html) to ./data

```
conda create -n de_cl python=3.11 pytorch=2.0.0
conda activate de_cl
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
run
```
python main.py preprocess config.yaml
python main.py train config.yaml
python main.py svm_decode config.yaml
python main.py knn_decode config.yaml
```
