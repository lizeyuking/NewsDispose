#coding=utf-8
import random
import kashgari
import jieba
import pandas as pd
import jieba.analyse

# 加载模型
loaded_model = kashgari.utils.load_model('cnn_classification_model')
#loaded_model.predict(random.sample(train_x, 10))

# # 预测指定样本
# news_sample = """
# 天灾无情，人有情”这句话，在中国永不过时！三天下了以往一年的雨，遭遇罕见强降水的郑州留下了无数惊心动魄而又感人的瞬间。截至发稿，名为“河南暴雨互助”的微博话题
# 阅读量已超28亿。昨（20日）晚开始，河南暴雨、尤其是降雨量一度达到1小时201.9毫米的郑州，就牵动着全国网友的心。很快，郑州消防员、武警河南总队官兵、中原火箭军、7省1800名消防员……他们都
# 来了！一夜过去，一条条网络视频和消息中，有险情、有驰援、有坚持，也有许多感动。暴雨来袭，20日多名群众浑身湿透一边大喊“抓紧”，一边紧紧拉住一根长绳，拼死拽上了在浑浊湍流中挣扎的女子。另一段视频中，商场工作人员和热心市民也是泡在齐腰的水里，
# 喊着号子，齐力逐个拉出负一楼的被困人员。郑州地铁里情况紧急，郑州消防迅速赶往事发地进行营救，被困者也齐心协力展开自救。
# 地铁5号线被困亲历者告诉@大象新闻：“大家都很团结，我们先采取自救再等待救援，妇女跟小孩先走，有劲的男士抬受伤的女士一块出来。”当晚，郑州图书馆主动发帖提醒附近被困人员，不要冒险回家，图书馆将提供无线网络、热水、简餐和休息场所。为了扩散援助消息，相关贴文已被转发近5万。不断有网友发布可供避险的场所、免费救援车辆信息等，其中有学校、企业、商户，也有个人。据@四川日报援引网友爆料，郑州酒店不仅没有涨价，还降价了。郑州高铁站内，从上海展演归来的师生捧起手中的乐器，为同样滞留车站的旅客演奏了《我和我的祖国》。21日早，央视记者从河南省委宣传部获悉，截至目前，洪灾已造成郑州市区12人死亡，当地已转移避险约10万人。部队、武警官兵、民兵、公安消防等多支救援力量投入救援，抢险仍在紧张进行。从“河南挺住”，到“河南一定行”，全国人民都在为河南加油"""
# x = list(jieba.cut(news_sample))
# y = loaded_model.predict([x])
# print(y[0]) # 输出游戏

f = open('数据集\\观察者网新闻.csv','r',encoding='utf-8') #导入要处理的原始数据文件
data = pd.read_csv(f).astype(str)
record_num = int(data.describe().iloc[0,0])
f.close()
data['分类预测'] = 0
#遍历每行review进行分词
for i in range(record_num):
    record = data.iloc[i,:]     #第i行的所有数据
    comment = record['review']  #标头为revie的数据
    x = list(jieba.cut(comment))
    result = loaded_model.predict([x])

    data.iloc[i,-1] = result[0]

data.to_csv("数据集\\观察者网新闻处理结果.csv") # 将新增的列数据，增加到原始数据中
