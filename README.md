# Neural Text Normalization

## 环境

```sh
conda create -n tn python==3.9
conda activate tn
pip install -r requirements.txt
```

## 模型和数据准备

使用的是[Text Normalization Challenge](https://www.kaggle.com/datasets/google-nlu/text-normalization)数据，下载后解压。

```shell
# 创建文件夹
mkdir dataset
# 解压到指定目录
uznip Google Text Normalization Challenge.zip -d dataset

# 下载模型权重google/electra-small-discriminator和google-t5/t5-small
mkdir models
chmod +x ./scripts/download.sh
./scripts/download.sh
```

生成训练数据

```python
# 这一步比较久
python upsample.py

python generate_decoder_dataset.py
```

## 训练

训练tagger
```shell
chmod +x ./scripts/run_tagger.sh
./scripts/run_tagger.sh
```


训练decoder
```shell
chmod +x ./scripts/run_decoder.sh
./scripts/run_decoder.sh
```