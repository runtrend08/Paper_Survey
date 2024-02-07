# 个人论文阅读总结
该文档为王攀依据个人阅读经历总结而成。论文选自2015年以后发表，围绕'**AI Enabled Network Traffic Classification**'为主题，所遴选出的代表性论文。目的是梳理该领域的学术研究方向，同时也为本团队的博/硕士研究生初学者所用。**版权所有 违者必究**
# 正文
## 第一阶段：萌芽阶段
这一阶段学者们尝试将深度学习（Deep Learning，DL）应用于NTC，主要围绕将各种经典的DL方法于NTC，评估其准确性，相较于经典的ML-TC。
### 1. 360公司王占一“开山之作”

 - **文章题目**：The Applications of Deep Learning on Traffic Identification
 - **发表单位**：360公司
 - **发表时间**：2015
 - **发表刊物**：Black Hat黑帽大会
 - **文章简介**：360的王占一发表了该领域的第一篇论文，首次提出用分组净荷类比为灰度图，然后采用MLP/CNN/SAE进行识别。
 - **面临挑战**：传统的DPI/ML都需要人工选择特征，耗时耗力，且无法识别未知应用；
 - **文章贡献**：**开山之作**。在该领域首次提出采用ANN/DL识别流量；并做了协议识别和未知流量识别两项实验工作。
 - **特征输入**：raw packets；
 - **模型训练**：有监督；MLP/CNN/SAE；
 - **数据集    **：私有。包括SSL/MySQL/SMB/SSH等；
 - **评估结果**：对协议识别较好，未知流量的识别一般。
 - **文章总结**：**GOOD**。AI TC 开山之作。
### 2. 中科院王伟 “2D-CNN+TC”

 - **文章题目**：Malware Traffic Classification Using Convolutional Neural Network for Representation Learning
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2017
 - **发表刊物**：2017 International Conference on Information Networking (ICOIN)
 - **文章简介**：从人工智能的角度提出了一个流量识别的分类法；然后提出了一个采用CNN的恶意流量分类方法，将流量数据当做图像（最佳分类器是Session+ALL）；训练了三个2D-CNN分类器：二分类、10分类和20分类器；
 - **面临挑战**：传统的DPI/ML都需要人工选择特征，耗时耗力，且无法识别未知应用；
 - **文章贡献**：**提升准确率**。专注用2D-CNN，从单向流Flow/会话Session，结合TCP/IP协议的全部层面ALL和仅第七层L7构建恶意流量识别模型。流量的预处理方法独特：先PCAP文件预处理，然后生成图像，再转为IDX文件用于CNN训练；
 - **特征输入**：raw packets（前784bytes）转图像（28*28）再转IDX文件；
 - **模型训练**：有监督；CNN（架构选用类LeNet-5）；因为输入是2维图像，所以2D-CNN；
 - **数据集	**：USTC-TFC2016,该数据集截取自CTU的Stratosphere IPS Project Dataset。包括Cridex等恶意流量；另外还有自己采的部分流量数据。
 - **评估结果**：与SOTA的IDS进行了对比，比如基于规则的snort和基于ML的Celik、Javaid等；
 - **文章总结**：**GOOD**。（1）没有加入时间相关的数据，只是采用的空间数据；（2）没有研究实际网络环境下流量大小是不固定的，模型的通用性有待提升，（3）只研究了已知的malware traffic，没有对未知流量识别进行研究。**该文章号称是第一篇用表示学习来识别Malware。**
### 3. 中科院王伟 “1D-CNN+TC”

 - **文章题目**：End-to-end encrypted traffic classification with one-dimensional convolution neural networks
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2017
 - **发表刊物**：2017 IEEE International Conference on Intelligence and Security Informatics (ISI)
 - **文章简介**：采用1D-CNN对ISCX数据集进行分类，他们认为流量字节一维序列数据，更类似自然语言，所以采用1D-CNN，经过试验，各项指标都超过了2D-CNN（前序文章成果）。
 - **面临挑战**：为提升TC识别精度；
 - **文章贡献**：**提升准确率**。提出“E2E TC”概念，构建1D-CNN TC；
  - **特征输入**：raw packets（前784bytes），作为串行数据，直接送模型，所以说是1D-CNN；
 - **模型训练**：有监督；CNN（架构选用类LeNet-5）；因为输入是2维图像，所以2D-CNN；
 - **数据集	**：ISCX 2016。
 - **评估结果**：与自己的前序工作2D-CNN以及C 4.5做了对比；
 - **文章总结**：**GOOD**。（1）净荷截断采用了固定的784字节，没有评估其他截断长度；（2）数据集不平衡；（3）non-VPN评估指标较差；

### 4 中科院王伟 “HAST-IDS”
 - **文章题目**：HAST-IDS: Learning Hierarchical Spatial-Temporal Features Using Deep Neural Networks to Improve Intrusion Detection
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2017.9
 - **发表刊物**：IEEE Access
 - **文章简介**：文章专注于Feature Learning，用CNN学习spatial features，用LSTM学习temporal features，作者提出了一个层次化的基于时空特征的IDS，名叫HAST-IDS。
 - **面临挑战**：特征工程自动化；
 - **文章贡献**：**特征提取**。（1）HAST-I：针对flow，选择其中的前N个BYTES，flow中多个分组首尾相接，然后截取前N个BYTES，送入CNN，通过Feature Learning，输出Flow Vector；（2）HAST-II：第一阶段，以packet为单位，取前q个BYTES，送CNN，形成PKT vector；第二阶段，多个PKT vector再送入LSTM输出Flow Vector；
  - **特征输入**：见上；
 - **模型训练**：有监督；CNN（架构选用类LeNet-5）；因为输入是2维图像，所以2D-CNN；
 - **数据集	**：ISCX 2012和DARPA99（因为这两个数据集有raw packets）。
 - **评估结果**：用t-SNE可视化输出的Flow vector；
 - **文章总结**：**GOOD**。给Features Learning提供了新思路，另外，采用**t-SNE可以有效的可视化Feature Learning的结果**，这为后面的Softmax分类铺垫了较好的基础；
![输入图片说明](/imgs/2024-01-29/UQRp5K5JAvvpr1bI.png)

### 5 中科院Bu，Zhiyong  “Network-in-Network”
 - **文章题目**：Encrypted Network Traffic Classification Using Deep and Parallel Network-in-Network Models
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2020
 - **发表刊物**：IEEE Access
 - **文章简介**：提出一种NIN（Network-in-Nework）的DL方法将固定长度的pkt vectors映射到分类标签，此外，还采用并行决策策略来构建两个子网络分别用于处理分组头部和分组净荷（主要的动机是考虑到二者携带着不同的流量特征）；
 - **面临挑战**：不同的特征输入携带不同的流量特征，单一NN无法解决，所以需要采用多个子网络；
 - **文章贡献**：N/A；
 - **数据集	**：ISCX 2012。
 - **评估结果**：与DeepPacket进行对比。实验结论得出分组头部的识别性能好，分组body很差。这个实验是存在重大问题的，因为分组头部的mac、IP和端口都不能作为input。；
 - **文章总结**：**BAD**。

### 6 中科院Cheng，Jin  “Lightweight ETC”
 - **文章题目**：Real-Time Encrypted Traffic Classification via Lightweight Neural Networks
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2020
 - **发表刊物**：IEEE GlobeCom
 - **文章简介**：基于 “最大化重用轻模块”的原则，采用多头注意力 1D-CNN这一“thin module”，来提取flow-全局特征和packet-局部特征；Multi-head attention允许一个pkt同时和多个后续同流的pkt交互，不同子空间的不同交互允许并行学习，极大降低参数数量和运行时间；采用ResNet作为基础模型进行训练。实验结果：准确率高于SOTA，参数数量为1D-CNN的1.8%；训练时间降低一半。
 - **面临挑战**：（1）DL所使用的的流量特征相对同质化；（2）大量模型都是用完整或大部分流做输入；（3）DL虽然准，但是牺牲计算资源换取准确率。
 - **文章贡献**：**轻量化、实时性**。（1）仅取一条流的first N （N=3）packets；（2）三个特征：分组位置、分组长度、该分组的头144bytes（去除了以太网头部。144包括20个字节的IP header、20个字节的TCP/UDP头部，当UDP时补12个zero、剩余104个净荷；超过104截断，不足补零）；性能评价指标除了准确率，还评估了**轻量相关的指标**。
 - **特征输入**：见上；注：去除了TCP的三次握手，因为与应用分类的特征无关；
 - **模型训练**：CNN（ResNet） + Multi-head Attention；
 - **数据集	**：ISCX 2012 VPN-nonVPN和HTTPS dataset。
 - **评估结果**：对比2个模型。（1）1D-CNN；（2）2D-CNN + LSTM。对比指标：**参数数量、权重数量、平均训练时间、平均测试时间；还做了时空复杂度分析**。
 - **文章总结**：**GOOD**。
![输入图片说明](/imgs/2024-01-31/XMZIbiu370MSlNOm.png)

### 7 中科院Huang，He  “Multi-task”
 - **文章题目**：Automatic Multi-task Learning System for Abnormal Network Traffic Detection
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2018
 - **发表刊物**：_International Journal of Emerging Technologies in Learning (iJET)_
 - **文章简介**：提出一种基于CNN的多任务学习系统，同步解决malware检测、VPN识别和木马分类的任务。
 - **面临挑战**：提高DL训练效率，类似“举一得三”。
 - **文章贡献**：**多任务**。
 - **特征输入**：raw packets转图像（32*32）；作者通过统计所有的分组长度，发现所有分组长度小于600，方差为78，因此他们认为定义32*32=1024可以cover99%的流量数据；
 - **模型训练**：CNN；
 - **数据集	**：CTU-13和ISCX（VPN），训练集：总共96000输入记录，每块流量数据是6000，（8个Trojan，4个VPN benign，4个non-VPN benign）；测试集：64000，每类4000；
 - **评估结果**：对比2个模型，一个HDCN；一个STL（单任务，模型自行构建）。另外对比了C4.5。
 - **文章总结**：**一般**。总体而言，是基于王伟的2D-CNN的文章思路，只是将输出变为了多任务。没有评估多任务训练和单任务训练在效率上的提升程度。
![输入图片说明](/imgs/2024-01-31/heUNQif9bd8hBUB0.png)

## 第二阶段
### 8 中科院Wang，Leiqi  “TGPrint”
 - **文章题目**：TGPrint: Attack fingerprint classification on encrypted network traffic based graph convolution attention networks
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2023.12
 - **发表刊物**：COSE: COMPUTERS & SECURITY
 - **文章简介**：提出一种基于graph of time-window叫做TGPrint的攻击指纹检测分类方法。先采用ML过滤掉normal中的无用、噪声数据；然后从恶意流量中创建描述攻击者-受害者主机交互行为的攻击图；此外，将每一个攻击划分特定的持续时间进一步修饰攻击图（包括时间、统计、聚合特征）；最后，采用GNN挖掘攻击图中的关键行为模式，从而生成指纹进而分类攻击（包括未知攻击）。
 - **面临挑战**：。
 - **文章贡献**：**攻击图、GNN**。
 - **特征输入**：
 - **模型训练**：；
 - **数据集	**：
 - **评估结果**：
 - **文章总结**：

### 9 中科院Yang，XD  “PETNet”
 - **文章题目**：PETNet: Plaintext-aware encrypted traffic detection network for identifying Cobalt Strike HTTPS traffics
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2024.1
 - **发表刊物**：COMNET: COMPUTERS NETWORKS
 - **文章简介**：Cobalt Strike（钴元素打击）是一种C&C攻击方式，常用语勒索软件中藏匿于HTTPS流量。文章提出了一种Plaintext-aware encrypted traffic detection network （PETNet）来识别该攻击。包括三个模块（1）Meta Info Modeling，也就是将握手净荷解析成具备语义的、与身份无关的meta特征；（2）序列化信息建模。通过Transformer Encoder建模攻击者-受害者之间的交互行为，通过meta信息指引下的注意力机制捕捉流量中多方面的相互关系，实现加密内容的配置可感知编码；（3）融合和预测。将互相关的meta信息与序列化信息相融合做最终的预测。
 - **面临挑战**：该攻击藏匿于HTTPS流量中，且通过伪造证书等骗过当前的常用检测手段，且现有手段很难获取多方面的信息以及流量中的相关性。
 - **文章贡献**：**流量语义化、Transformer编码、配置可感知、可抵御概念漂移**。
 - **特征输入**：
 - **模型训练**：；
 - **数据集	**：在close-world和open-world数据集上进行评估。
 - **评估结果**：平均提高53.42%，14个月时间内容可以有效环节概念漂移问题。证明其有效性和泛化能力。
 - **文章总结**：

### 10 中科院Yua，QJ  “MCRe”
 - **文章题目**：MCRe: A Unified Framework for Handling Malicious Traffic With Noise Labels Based on Multidimensional Constraint Representation
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2024.1
 - **发表刊物**：IEEE TIFS
 - **文章简介**：文章提出一种基于多维约束表达（MCRe）的通用框架，用于解决恶意流量数据集中的噪声标记问题。其将data cleaning和robust training统一到一个理想的表达函数逼近中，其包含几个特质：信息完整性约束、簇分离约束和核心逼近约束，被定义用来驱动MCRe在迭代中逼近最理想的表达函数。这些约束让MCRe可以学到单独、类内、全局的分布式知识，避免不合逻辑的领域知识提取，确保强噪声鲁棒性。
 - **面临挑战**：真实恶意流量数据集中“标记噪声”问题很严重，直接影响到基于DL的IDS的有效性。处理噪声标记的数据集现有方法主要分为两类：（1）data cleaning；（2）robust training；但是这两种方法会忽略数据集中不同组件的信息，导致在高噪声环境下性能的断崖式下滑（Cliff-like drop）。
 - **文章贡献**：**流量语义化、Transformer编码、配置可感知、可抵御概念漂移**。
 - **特征输入**：
 - **模型训练**：；
 - **数据集	**：在22类真实恶意流量数据集上。
 - **评估结果**：平均提高53.42%，14个月时间内容可以有效环节概念漂移问题。证明其有效性和泛化能力。
 - **文章总结**：

### 11 中科院Li，Xiang  “Let Model Keep Evolving”
 - **文章题目**：Let model keep evolving: Incremental learning for encrypted traffic classification
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2023.12
 - **发表刊物**：COSE
 - **文章简介**：文章提出一种基于多视角融合MISS方法的增量学习IL框架，使得ETC模型可持续。MISS的新颖性在于：（1）从多个视角的流量序列中提取跨视角信息来捕捉足够的知识；（2）提出了一个标本选择算法降低数据集的存储消耗；（3）从参数的可学习性方面设计了一组分支来缓解演进过程中的准确率损失；主要有几个关键步骤：1.代表性的标本选择，重组新的数据集；2.
 - **面临挑战**：网络环境不断变化和演进，新应用不断出现，使得原本训练好的ETC模型逐渐失效。传统的方法是在新数据集（新老应用数据）上重新训练，然而局限性在于算力和存储无效浪费。**应用于CV中的传统IL方法**：主要考虑如何避免灾难性遗忘。（1）采用固定长度的存储空间存放标本，用于后期重放以避免遗忘；（2）合成老任务的样本来平衡新旧任务的标本数量；（3）通过设计渐进性网络来保障不同的增量阶段中部分恒定不变的模型参数；（4）引入约束条件，来缓解新老任务的数据不均衡问题；（5）在知识蒸馏中使用可变的池化维度用于在后续模型之间桥接Feature表达；（6）在每个增量步骤中，通过增加额外的新网络来学习新特征，在分类任务中使用首尾相接的新老features；（7）类增量学习，使用知识蒸馏结合适应性的特征固化来防止遗忘老知识；**应用于TC的IL方法**：（1）One vs Rest策略；当引入大量新应用时，扩展性较差；（2）采用iCarl框架；（3）使用知识蒸馏结合bias修正；（4）多类SVM；（5）增量SVM+SGD；这篇（6）联邦增量学习；这篇文章认为以往的研究主要集中于将传统领域（如CV）的IL直接应用在ETC上，但不适用，需要更多研究features。
 - **文章贡献**：**增量学习**。
 - **特征输入**：（1）文章提出了Multi-View，我理解是指代四类view，分别是包间时间间隔IAT、净荷序列、分组长度序列和内容类型序列。如Table 1。（2）一个流选择了first n packets，N = 32；每个分组选择了前128个字节的内容；（3）具体的Multi-view见Fig.2。我理解其实就是从四类特征（所谓的四个视角）构建模型的特征输入；
 - **模型训练**：；
 - **数据集	**：在22类真实恶意流量数据集上。
 - **评估结果**：平均提高53.42%，14个月时间内容可以有效环节概念漂移问题。证明其有效性和泛化能力。
 - **文章总结**：
![输入图片说明](/imgs/2024-02-01/uV8BVM600fPPNife.png)
![输入图片说明](/imgs/2024-02-01/FnBujWfqSXg9bmER.png)
![输入图片说明](/imgs/2024-02-01/ZwvMMEwY7d3HBYq5.png)


### 12 中科院Lin，Xinjie “CETP”
 - **文章题目**：CETP: A Novel Semi-supervised Framework based on Contrastive Pre-training for Imbalanced Encrypted Traffic Classification Representation
 - **发表单位**：中科院；中国科技大学
 - **发表时间**：2024.1
 - **发表刊物**：COSE（Review）
 - **文章简介**：针对真实网络环境下，（1）流量类别不平衡导致的Label Bias，（2）流量同质化；如何利用广泛易获取到的无监督流量数据学习强泛化能力的数据表达依然是一个很大的挑战。文章提出一种类别不平衡流量表达模型，叫做Contrastive Encrypted Traffic Pre-training（CETP）。该模型通过半监督持续调优、伪标记递归和动态Loss-weighting 算法来缓解label bias。
 - **面临挑战**：**问题背景**：针对真实网络环境下，（1）流量类别不平衡导致的Label Bias，（2）流量同质化（指代同一个APP厂商家族旗下会有多个APP，比如腾讯、阿里，这些APP会共用开发组件，比如认证、广告、分析等，不同APP应用会产生类似的加密流量，易造成流量同质化）；如何利用广泛易获取到的无监督流量数据学习强泛化能力的数据表达依然是一个很大的挑战。**现存技术局限性**：（1）数据再平衡：Oversampling/SMOTE/GAN，依赖于数据分布的先验知识，缺乏泛化性；（2）改造Loss Function。比如Focal Loss，Cost-Sensitive；在样本极端不平衡条件下存在局限性。（3）Pre-training：利用大量无标记数据来获取无偏差的数据表达，是一个新的方向。本文就是Motivated by this。
 - **文章贡献**：**类别不平衡、半监督、伪标签、Transformer**。（1）CETP：为了解决不平衡流量带来的Label Bias问题，通过Pre-Training，不直接用类标记，从无标记数据中学习鲁棒的流量表达；（2）基于无监督数据，精心构建了正负对，并学习通用的流量表达；（3）应用两个预训练任务（CLM，Contrastive Learning Model）和（MSM，Masked Sequence Model）来构建正负对采样方法，目的是捕捉流量上下文关系特征，构建通用的流量表达；（4）利用伪标记递归、动态Loss weighting方法做半监督调优。
 - **特征输入**：原始报文RD（Raw Datagram）和统计特征（包长度，PL）。每个流取前32个分组、128个字节。每个字节序列由2个字节构成一个unit，用Byte-Pair Encoding，长度用‘+、-’标明方向。
 - **模型训练**：；
 - **数据集	**：CP-Android、CIC-2019、ISCX-VPN。
 - **评估结果**：平均提高53.42%，14个月时间内容可以有效环节概念漂移问题。证明其有效性和泛化能力。
 - **文章总结**：

<!--stackedit_data:
eyJoaXN0b3J5IjpbODQ3NTQ1OTE1LDEwNzgwNjQ2NjYsLTIwOT
AzODIwNDIsLTE1MTY3MzE0OTUsLTE3MDE3ODA2MTcsLTE0MTYz
NDQwNjQsODY5NDQwNDE2LDExNjc1MTc4ODMsMjA5MTA4OTUxMS
wxNTQ5NDU0NTQzLDEwNDU2Mjc5MTksOTMwNzc2ODE5XX0=
-->