# 中文新词发现 - 改进版

Python3利用互信息和左右信息熵的中文分词新词发现

原作者：[zhanzecheng](https://github.com/zhanzecheng)

改写：[DrDavidS](https://github.com/DrDavidS/Chinese_segment_augment)

## 简介

* 使用[jieba分词](https://github.com/fxsjy/jieba)为基本分词组件
* 针对用户给出的文本，利用信息熵进行新词发现
* 使用[字典树](https://github.com/zhanzecheng/The-Art-Of-Programming-By-July/blob/master/ebook/zh/06.09.md)存储单词和统计词频
* 由于但文本不能正确反映单个词的词频，这里使用[jieba](https://github.com/fxsjy/jieba)自带的词频表作为外部数据源
* 取 TOP N 个作为新词

## 改进

* 修改了旧代码在windows上的编码报错问题。
* 修改了旧代码执行极其缓慢的问题，采用字典代替列表。
* 新增结果保存，以 txt 形式存放在 `result_new_words` 文件夹中，默认保存得分前100的新词。根据语料大小可以在 `demo_run.py` 中自行更改topN。

## 使用配置

    git clone https://github.com/DrDavidS/Chinese_segment_augment.git
    pip3 install jieba

## 使用方式

直接运行：

    python demo_run.py  

飞一般地感觉。

> 使用你自己的语料：
>
> 在 `data` 文件夹中的 `demo.txt` 中，按格式替换为你自己的语料即可。

## 得到 TOP N 得分的新词

    # result里面存储的是所有新词和其得分，add_word里面是top100
    result, add_word = root.wordFind(100)

具体细节请参考 `demo_run.py`。

## 效果说明

初始语句：

    蔡英文在昨天应民进党当局的邀请，准备和陈时中一道前往世界卫生大会，和谈有关九二共识问题

添加前：

    蔡/ 英文/ 在/ 昨天/ 应/ 民进党/ 当局/ 邀请/ 准备/ 和/ 陈时/ 中/ 一道/ 前往/ 世界卫生/ 大会/ 和谈/ 有关/ 九二/ 共识/ 问题/ 
添加后：

    蔡英文/ 在/ 昨天/ 应/ 民进党当局/ 邀请/ 准备/ 和/ 陈时中/ 一道/ 前往/ 世界卫生大会/ 和谈/ 有关/ 九二共识/ 问题/

新词结果和得分（参考）：

    世界卫生大会 ---->   0.4380419441616299
    蔡英文      ---->   0.28882968751888893
    民进党当局   ---->   0.2247420989996931
    陈时中      ---->   0.15996145099751344
    九二共识    ---->   0.14723726297223602

测试样本：

    台湾“中时电子报”26日报道称，蔡英文今日一早会见“世卫行动团”，她称，台湾虽然无法参加WHA(世界卫生大会)，但“还是要有贡献”。于是，她表示要捐100万美元给WHO对抗埃博拉病毒
    对于台湾为何不能，蔡英文又一次惯性“甩锅”，宣称“中国对台湾的外交打压已无所不用其极”。
    ......

保存在 demo.txt中。

## 方法解释

* 先使用jieba分词对demo.txt做粗略分词
* 使用 3-gram 的方式来构建节点，并使用词典树对存储分词，如

        [4G, 网络， 上网卡] --> [4G, 网络， 上网卡, 4G网络, 网络上网卡, 4G网络上网卡]

* 利用trie树计算互信息 PMI
* 利用trie树计算左右熵
* 得出得分 score = PMI + min(左熵， 右熵)
* 以得分高低进行排序，取出前5个，若前面的待选词在属于后面待选词一部分，则删除后面待选词，如

        [花呗， 蚂蚁花呗] --> [花呗]

> 具体原理说明请看这个[链接](https://www.jianshu.com/p/e9313fd692ef)

## 补充说明

* 3-gram 和 4-gram 或者 5 - gram 的效果区别？
  * 并不能做 4-gram，因为这里是词的gram不是字的gram。
  * 5-gram 同理，太长了没必要。

* 标点符号对于效果的影响？
  * 建议清洗标点符号，尝试过一次不清洗直接放进去，发现会报一些谜之错误。
  * 可以保留逗号`，`，句号`。`，分号`；`等常用标点。
  * 数字清洗效果比较难说，可能有帮助，也可能帮倒忙。

* 分句和长句对效果的影响？
  * 似乎没有区别。

* 原作代码以 jieba 切分出的词代替字，所以用的 3-gram 足矣，如果有需要可以以字分隔改改试试，这个gram可以取大一些。

* 互信息的计算会受到语料本身的影响，有时候小语料中能识别的新词，扩大语料范围就不能识别了，比较尴尬。
  * 换句话说，识别出的新词最后走一次人工筛选，工作量减小很多。但是漏网的新词就比较尴尬了，recall不足。
  * 考虑增加topN，或者给计算公式改进一下，比如加点权重因子等。
