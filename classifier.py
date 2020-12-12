import os
import torch
import torch.nn.functional as F
import pandas as pd

from transformers import BertForSequenceClassification, BertConfig
from transformers import BertTokenizer

class SentimentClassifier():
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
        self.model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config)
        self.model.load_state_dict(torch.load(os.path.join("model", "model_sentiment.bin")))
        self.results = list()

    def predict(self,text):
        i2w = {0: 'positive', 1: 'neutral', 2: 'negative'}
        text = text[:512]
        subwords = self.tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1,-1).to(self.model.device)

        logits = self.model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        return i2w[label] , str(f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}')
    
    def process_text(self,text):
        result = dict()
        label , conf = self.predict(text=text)
        print(f"Success Predict {label}, {conf}")
        # result['text'] = text
        result['label'] = label
        result['conf'] = conf
        self.results.append(result)
        # print(self.results)
    
    def get_result(self):
        return self.results
    def reset(self):
        self.results = list()

class CategoryClassifier():
    
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        config = BertConfig.from_pretrained("indobenchmark/indobert-base-p1", num_labels=3)
        self.model = BertForSequenceClassification.from_pretrained("indobenchmark/indobert-base-p1", config=config)
        self.model.load_state_dict(torch.load(os.path.join("model", "model_category.bin")))
        self.results = list()

    def predict(self,text):
        i2w = {0: 'tekno', 1: 'health', 2: 'business'}
        text = text[:512]
        subwords = self.tokenizer.encode(text)
        subwords = torch.LongTensor(subwords).view(1,-1).to(self.model.device)

        logits = self.model(subwords)[0]
        label = torch.topk(logits, k=1, dim=-1)[1].squeeze().item()

        return i2w[label] , str(f'{F.softmax(logits, dim=-1).squeeze()[label] * 100:.2f}')
    
    def process_text(self,text):
        result = dict()
        label , conf = self.predict(text=text)
        print(f"Success Predict {label}, {conf}")
        # result['text'] = text
        result['label'] = label
        result['conf'] = conf
        self.results.append(result)
        # print(self.results)
    
    def get_result(self):
        return self.results
    def reset(self):
        self.results = list()

class Classifier():
    def __init__(self):
        self.category_classifier = CategoryClassifier()
        self.sentiment_classifier = SentimentClassifier()

    def process(self,text):
        self.category_classifier.process_text(text)
        self.sentiment_classifier.process_text(text)
        return dict(category=self.category_classifier.get_result(), sentiment=self.sentiment_classifier.get_result())

if __name__ == "__main__":
    clf = Classifier()
    input_string = """
    Menteri Ketenagakerjaan Ida Fauziyah mengatakan, untuk melindungi   Pekerja  Migran Indonesia (PMI), kuncinya adalah sinergitas dan kolaborasi seluruh pihak. 
 Untuk itu, Kementerian Ketenagakerjaan (Kemnaker) bersama Kepolisian Republik Indonesia (Polri) menandatangani   nota    kesepahaman  yang memuat kesinergisan pelaksana tugas dan fungsi ketenagakerjaan di Jakarta, Kamis (19/11/2020). 
 â€œSaya mengajak kita semua untuk semakin  aware  terhadap perlindungan kepada   pekerja  migran kita, karena kita tidak bisa menunda, meniadakan bekerja ke luar negeri, karena itu adalah hak warga negara yang dilindungi konstitusi,â€ ujarnya. 
 Sementara itu, kata Ida,Â memberikan pelindungan kepada pekerja migran adalah kewajibanÂ  negara atau pemerintah. 
 Ida menjelaskan, sinergitas Kemnaker-Polri ini merupakan bentuk komitmen kedua institusi untuk memperkuat perlindungan bagi pekerja migran. 
 
 Selain tugas dan fungsi, penguatan sinergitas tersebut juga mencakup pertukaran data atau informasi dan pendampingan dalam penanganan Calon PMI non-prosedural. 
 Ida juga menyatakan, saat ini Indonesia memiliki regulasi yang baik dalam hal penempatan dan   perlindungan pekerja  migran. 
 Regulasi tersebut adalah Undang-Undang Nomor 18 Tahun 2017 tentang   Perlindungan Pekerja  Migran (UU PPMI). 
 Sebagaimana diamanatkan UU PPMI, lanjutnya, perlindungan pekerja migran melibatkan berbagai elemen, baik di pusat maupun di daerah. 
 Dia mengatakan, perlindungan di UU Nomor 18 dilakukan dari hulu sampai hilir, dari kampung halaman sampai kembali ke kampung halaman. Begitu juga tugas dan fungsi  stakeholder . 
 
 â€œYang dibutuhkan sekarang sinergitas, koordinasi antar  stakeholder , termasuk sinergi dengan Kepolisian Republik Indonesia,â€ katanya seperti keterangan tertulis yang diterima Kompas.com. 
 Adapun,   nota kesepahaman  antara Kemnaker dan Polri ini bertujuan untuk memperkuat koordinasi dan kolaborasi dalam menyinergikan tugas dan fungsi Kemnaker dengan Polri. 
 Ruang lingkup   Nota    Kesepahaman  ini, antara lain pertukaran data dan/atau informasi; pencegahan, penanganan, dan penegakan hukum; bantuan pengamanan; peningkatan kapasitas dan pemanfaatan sumber daya manusia; pemanfaatan sarana dan prasarana; dan kegiatan lain yang disepakati. 
 Selain nota kesepahaman, pada kesempatan ini juga ditandatangani Perjanjian Kerja Bersama (PKB) tentang Pertukaran Data dan/atau Informasi serta Pendampingan dalam Penanganan Penempatan Calon Pekerja Migran Indonesia atau Pekerja Migran Indonesia Yang tidak Sesuai Prosedur. 
 
 PKB ini ditandatangani Direktur Jenderal Pembinaan Penempatan Tenaga Kerja dan Perluasan Kesempatan Kerja (Binapenta dan PKK) dan Kepala Badan Reserse Kriminal Polri. 
 PKB ini merupakan tindak lanjut dari salah satu ruang lingkup   Nota Kesepahaman , khususnya terkait penanganan penempatan Pekerja Migran Indonesia yang tidak sesuai prosedur. PKB ini akan berlaku selama 5 tahun. 
 Untuk mendukung implementasi nota kesepahaman ini, Menaker Ida juga meminta Polri menyosialisasikan kerja sama ini kepada jajarannya di daerah. 
 â€œSaya juga minta kepada Ditjen Binapenta dan PKK untuk menyosialisasikan kerja sama ini hingga ke daerah,â€ katanya. 
 Sementara itu, Wakil Kepala Polri Gatot Eddy Pramono menuturkan, pihaknya mengapresiasi nota kesepaham dan PKB yang telah ditandatangani. 
 
 Dia juga mendukung penuh agar nota kesepahaman dan PKB ini dapat terimplementasi dengan baik. 
 Bahkan, pihaknya siap membantu dalam penyiapan kompetensi calon pekerja migran, salah satunya dengan dukungan sarana dan prasarana. 
 â€œUntuk sarana dan prasarana yang dibutuhkan, Polri siap membantu, ini hal-hal yang harus kita laksanakan,â€ terangnya. 
 Terkait nota kesepahaman, dia juga mengajak seluruh institusi yang ada di kepolisian menyosialisasikannya
    """
    result = clf.process(input_string)
    print(result)