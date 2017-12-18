from gensim.models import word2vec
import logging
import sys

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

pref = ['北海道',
        '青森県','岩手県','宮城県','秋田県','山形県','福島県',
        '東京都','神奈川県','埼玉県','千葉県','茨城県','栃木県','群馬県',
        '山梨県','新潟県','長野県',
        '愛知県','岐阜県','静岡県','富山県','石川県','福井県',
        '三重県','大阪府','兵庫県','京都府','滋賀県','奈良県','和歌山県',
        '島根県','鳥取県','岡山県','広島県','山口県',
        '徳島県','香川県','愛媛県','高知県',
        '福岡県','佐賀県','長崎県','熊本県','大分県','宮崎県','鹿児島県',
        '沖縄県']
model = word2vec.Word2Vec.load(sys.argv[1])
cosSimPref = []
for i in pref:
	cosSim = []
	cosSim.append(i)
	cosSim.append(model.similarity(sys.argv[2], i))
	cosSimPref.append(cosSim)

cosSimPref.sort(key=lambda x:x[1])
cosSimPref.reverse()

for i in cosSimPref:
	print(i[0], i[1])

 
