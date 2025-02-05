<img src="./f-images/word embedding demo.jpg">

## 1. Tokenizer
如果大家用过 huggingface，对 tokenizer 肯定不会陌生。在使用模型前，都需要将sequence 过一遍 tokenizer，进去的是 word 序列（句子），出来的是 number 序列。但是，HF 的 tokenizer 到底在做什么？事实上，tokenizer 总体上做三件事情：
1. **分词:** tokenizer将字符串分为一些 sub-word token string，再将 token string 映射到 id，并保留来回映射的 mapping。从 string 映射到 id 为 tokenizer encode 过程，从 id 映射回 token 为 tokenizer decode 过程。映射方法有多种，例如 BERT 用的是 WordPiece，GPT-2 和 RoBERTa 用的是 BPE 等等，后面会详细介绍。
2. **扩展词汇表:** 部分 tokenizer 会用一种统一的方法将训练语料出现的且词汇表中本来没有的 token 加入词汇表。对于不支持的 tokenizer，用户也可以手动添加。
3. **识别并处理特殊token:=** 特殊 token 包括 [MASK], \<|im_start|\>, \<sos\>, \<s\> 等等。tokenizer 会将它们加入词汇表中，并且保证它们在模型中不被切成 sub-word，而是完整保留。

### 1.1 分词粒度
我们首先来看一下几种不同的分词粒度。最直观的分词是**单词分词法（word base）**。单词分词法将一个 word 作为最小单元，也就是根据空格或者标点分词。拿大模型中的分词器 tokenizer 一文中的例子来说，Today is Sunday 用 word-base 来进行分词会变成['Today', 'is', 'Sunday']。

最详尽的分词是**单字分词法（character-base）**。单字分词法会穷举所有出现的字符，所以是最完整的。在上面的例子中，单字分词法会生成['T', 'o', 'd', ..., 'a', 'y']。

另外还有一种最常用的、介于两种方法之间的分词法叫**子词分词法**，会把上面的句子分成最小可分的子词['To', 'day', 'is', 'S', 'un', 'day']。子词分词法有很多不同取得最小可分子词的方法，例如BPE（Byte-Pair Encoding，字节对编码法），WordPiece，SentencePiece，Unigram等等。接下来我们具体看看各大主流模型用的是什么分词法。

## 2. Embedding Layer
tokenize 完的下一步就是将 token 的 one-hot 编码转换成更 dense 的 embedding 编码。在 ELMo 之前的模型中，embedding 模型很多是单独训练的，而 ELMo 之后则爆发了直接将 embedding 层和上面的语言模型层共同训练的浪潮（ELMo的全名就是Embeddings from Language Model）。不管是哪种方法，Embedding 层的形状都是一样的。我们举个例子来看看 embedding 层是怎么工作的。在 huggingface 中，seq2seq模型往往是这样调用的：
```
input_ids = tokenizer.encode('Hello World!', return_tensors='pt')
output = model.generate(input_ids, max_length=50)
tokenizer.decode(output[0])
```
上面的代码主要涉及三个操作：tokenizer将输入encode成数字输入给模型，模型generate出输出数字输入给tokenizer，tokenizer将输出数字decode成token并返回。

例如，如果我们使用T5 TokenizerFast 来 tokenize 'Hello World!'，则：
> 1. tokenizer会将token序列 ['Hello', 'World', '!'] 编码成数字序列[8774, 1150, 55, 1]，也就是['Hello', 'World', '!', '</s>']，然后在句尾加一个</s>表示句子结束。
> 2. 这四个数字会变成四个one-hot向量，例如8774会变成[0, 0, ..., 1, 0, 0..., 0, 0]，其中向量的index为8774的位置为1，其他位置全部为0。假设词表里面一共有30k个可能出现的token，则向量长度也是30k，这样才能保证出现的每个单词都能被one-hot向量表示。
> 3. 也就是说，一个形状为 (4) 的输入序列向量，会变成形状为 (4, 30k) 的输入 one-hot 向量。为了将每个单词转换为一个 word embedding，每个向量都需要被被送到embedding 层进行dense降维。
> 4. 现在思考一下，多大的矩阵可以满足这个要求？没错，假设embedding size为768，则矩阵的形状应该为 (30k, 768)，与BERT的实现一致：
<img src="./f-images/code demo.jpg">

## 3. 各路语言模型中的tokenizer
我整理了一下各大LM用的tokenizer和对应的词汇表大小：
|LM|Tokenizer|Vocabulary Size|
|--|--|--|
|BERT|	Word-Piece|	30k|
|ALBERT|	Sentence-Piece|	30k|
|RoBERTa|	BPE|	50k|
|XLM-RoBERTa|	Sentence-Piece|	30k|
|GPT|	SpaCy|	40k|
|GPT-2|	BPE|	50k|
|GPT-3|	BPE|	50k|
|GPT-3.5 (ChatGPT)|	BPE|	-|
|GPT-4|	BPE|	-|
|T5|	Sentence-Piece|	30k|
|Flan T5|	Sentence-Piece|	30k|
|BART|	Word-Piece|	50k|

另外，有大佬做了各大LLM的词汇表大小和性能：
|名称|词表长度↑|中文平均长度↓|英文平均长度↓|中文处理时间↓|英文处理时间↓|
|--|--|--|--|--|--|
|LLaMA|	32000|	62.8|	32.8|	02:09|	01:37|
|BELLE|	79458|	24.3|	32.1|	00:52|	01:27|
|MOSS|	106072|	24.8|	28.3|	07:08|	00:49|
|GPT4|	50281|	49.9|	27.1|	00:07|	00:08|
|BLOOM/Z|	250680|	23.4|	27.6|	00:46|	01:00|
|ChatGLM|	130344|	23.6|	28.7|	00:26|	00:39|

## 参考资料
* [从词到数：Tokenizer与Embedding串讲](https://zhuanlan.zhihu.com/p/631463712)