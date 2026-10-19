# RoleHOS 可复现研究代码

代码发布于GitHub，权重通过Google Drive独立提供。包含最终论文对应的 12 个配置×3 随机种子，共36组权重、配置、原训练/评估日志、推理/评估/训练脚本和 Gradio 演示。

## 代码与权重下载

- [GitHub代码仓库](https://github.com/ggoodd-any/RoleHOS)
- [Google Drive权重目录：12种配置、每种3个种子](https://drive.google.com/drive/folders/1qylFVDv40EpEyo6JSRhAhpcej5M-VG0h)
- [权重包SHA-256清单](weights/downloads.json)

运行主模型或Gradio只需下载 `RoleHOS_weights_full.zip`；消融配置按需下载对应ZIP。在仓库根目录按下列命令提取模型文件，保留 `weights/<配置>/` 结构和仓库中的最新说明。压缩包另附冻结的清单与许可文件：

```bash
git clone https://github.com/ggoodd-any/RoleHOS.git
cd RoleHOS
unzip /path/to/RoleHOS_weights_full.zip 'weights/*/*.pth' -d .
```

公开仓库不再分发数据集原图。可直接上传自己的RGB图像，或取得数据集后运行 `python -m tools.import_examples --data-root /path/to/egohos` 恢复六张精选样图。

## 安装与演示

原训练环境：Linux、单张 RTX 5090、Python 3.10.20、PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128、CUDA 12.8、MMEngine 0.10.7、MMCV 2.2.0、NumPy 1.26.4、OpenCV 4.11.0.86。本精简版本使用 MMCV-lite，不需要编译额外 CUDA 算子或安装 MMDetection；Gradio 使用 Pillow 11.3.0，原训练为9.5.0，模型前处理均通过 OpenCV 完成。完整依赖见 `requirements*.txt`。

```bash
conda create -n rolehos python=3.10.20 -y
conda activate rolehos
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-demo.txt -c constraints-tested.txt
python -m unittest discover -s tests
python -m tools.verify_assets
bash scripts/demo.sh --device cpu
```

浏览器打开 http://127.0.0.1:7860 ，选示例或上传图片，选择seed，点击运行。支持查看叠加图、彩色角色图，下载0–5类别编号的PNG。六张样图分别来自ID/OOD/Mini，每组两张：先从每组64张均匀抽取的候选图中按seed 0预测与标注的重叠程度筛选，再目视检查并兼顾场景及角色多样性。这是分割效果较好的精选展示，不代表完整测试集表现，也不改变论文评估指标；来源、筛选依据和哈希见 `examples/manifest.json`。界面只在本机监听，不启用公开分享。这里运行的是PyTorch FP32模型，界面响应时间不等于论文的Jetson FPS。

## 一键训练、测试、消融

首先按 `docs/DATA.md` 获取对应的六类标签版本，并取得官方DINOv3 ConvNeXt-Tiny LVD-1689M初始化权重。当前已整理的RoleHOS最终权重用于测试，不能替代从预训练主干开始训练。

```bash
python -m tools.check_data --data-root /path/to/egohos
bash scripts/train.sh --data-root /path/to/egohos --pretrained /path/to/dinov3_convnext_tiny_pretrain_lvd1689m-21b726bb.pth --seed 0
bash scripts/test.sh --data-root /path/to/egohos --seed 0 --save-masks --output outputs/full_seed0
bash scripts/test_ablations.sh --data-root /path/to/egohos
# 如需依次重新训练全部消融：
bash scripts/test_ablations.sh --mode train --data-root /path/to/egohos --pretrained /path/to/dinov3_initialization.pth
```

在目录根部运行；shell脚本会自动定位根目录。`PYTHON=/path/to/python`可指定解释器。`--variants`和`--seeds`可筛选消融。`--limit 2`仅检查小样本，结果标为partial、不报告全量Avg。训练可用`--smoke-iters 2 --device cpu`检查两步流程，但会使用batch1，不是正式复现实验。

## 关键协议

- 固定种子0/1/2；FP32训练20,000次迭代，batch8，AdamW学习率6e-5、主干倍率0.2、weight decay0.01、梯度裁剪5；1,000步线性预热，再多项式衰减。评估最终20k检查点，不挑测试最优。
- 输入448×448 RGB；均值`[106.01075,95.40013,87.42854]`，标准差`[64.356636,60.888744,61.41911]`。训练水平翻转同时交换LH/RH、LO/RO，TO不变。精确损失权重和其他设置保留在配置文件。
- ID/OOD/Mini分别1126/500/1095张，合计2721张；验证1124张单列。由累计六类混淆矩阵计算前景mIoU；Mini平均LH/RH/RO/TO，其他两组平均五前景。Avg为三个测试集mIoU的算术平均，验证集不参与。论文主模型72.54%是三种子均值。
- 推理开启PyTorch严格确定性算法、设置Python/NumPy/Torch/CUDA种子、关闭cuDNN benchmark和TF32，设置CUBLAS工作区。原训练采用cuDNN确定性模式、保留cuDNN TF32默认值，并未启用全局严格算法；本包如实区分。不同硬件和软件版本仍可能有边界像素差异。
- 新训练日志、完整优化器检查点写入`outputs/`；支持显式`--resume`，但不承诺与历史中断实验完全相同的数据顺序。重新从头训练是标准复现方式。本轮仅做接口、数值回归和短步训练检查，未重新训练36个完整实验。

## 权重与日志

权重结构为`weights/<配置>/rolehos_<配置>_seedN_iter20000.pth`，可读文件名，纯state_dict格式，未带优化器或本机路径。全部参数与原检查点逐张量核对。`weights/manifest.json`保存新旧SHA256、配置、原始指标和日志关联。最终论文的无SFSC/无模块消融使用零上下文版本，避免混入较早同名实验。

`logs/reference/`保存脱敏原训练log、scalar日志和评估摘要；保留数值与种子，替换本机绝对路径，未改动原日志。`logs/release_checks/`是本轮包装验证证据。`docs/ABLATIONS.md`与`expected_metrics.csv`给出全部对应关系。

```bash
python -m tools.package_release --output /path/to/bundles
```

打包代码和按配置分组的权重ZIP，供之后手动上传Google Drive，不会访问账户或自动发布。公开代码包默认不包含数据集图像，授权数据用户可运行`python -m tools.import_examples --data-root ...`恢复相同样图；本地审阅目录已附样图。

## 许可

新增自有代码按MIT；第三方Apache、FAIR/EUPE许可及DINOv3权重许可分别保留，不能把整个权重/代码/数据包统称为MIT。具体见`NOTICE.md`和`licenses/`。样图遵守原数据集条款，正式公开前需核准再分发范围。本包没有替换模型来规避这些条款。代码与权重下载入口见上方。
