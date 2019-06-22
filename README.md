# ChieseWordSegment
 Bi-LSTM + CRF  Model

  输入句子由80个词组成，每个词由100维的词向量表示，则模型对应的输入是（80，100），经过BiLSTM后隐层向量变为T1（80，128），
其中128为模型中BiLSTM的输出维度。Dropout层防止过拟合，再经过TimeDistribute层将128维对应为5维。设分词任务的目标标签为
X（0）,B（Begin）,M（Middle）,E（End）,S（Single），使用CRF层模型最终输出维度为（120，5）的向量。
