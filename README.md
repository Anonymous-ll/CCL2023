# CCL2023 源代码与数据集实现步骤说明

网站中提供的是在CCL2023中提交的论文“基于自监督学习的方面聚类”的源代码与数据集文件。请根据以下步骤实现模型的训练：

(1) unzip the file amazon-review-100k.rar

(2) run the file hrea_amazon_datapreprocess.py to process dataset.

(3) run the file hrea_amazon_encoder_train.py to train a text autoencoder.

(4) run the file hrea_amazon_encoder_visual.ipynb to see the visulized reviews

(5) run the file hrea_amazon_level1_train.py to train the first student model

(6) run the file hrea_amazon_level1_explore.py to explore the aspect-groups in first level

# 数据集说明
amazon-aspects-category-distribution.txt：方面的类别分布文档
amazon-review-100k.rar：Amazon数据集文档
amazon-embed.txt：Amazon数据集词嵌入文档

# 使用t-SNE可视化的实验效果图
![visual](https://user-images.githubusercontent.com/130581857/231504050-030e2eb6-03ca-4aae-a3f4-3c536ee36d68.png)
