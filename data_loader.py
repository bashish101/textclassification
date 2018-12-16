import os
import re
import numpy as np
import pandas as pd
from collections import Counter

class DataLoader(object):
	def __init__(self,
		     data_path = "train",
		     val_data_path = "test",
		     stopwords_path = "stopwords.txt",
		     vocab_path = "vocab.txt",
		     max_vocab_len = 20000,
		     max_sequence_len = 400,
		     train_split = 0.8,
		     preprocess_prefix = 'preprocessed_data',
		     classes_path = "classes.txt",
		     ignore_class = 'ignore'):
		self.data_path = data_path
		self.val_data_path = val_data_path
		self.stopwords_path = stopwords_path
		self.vocab_path = vocab_path

		self.preprocess_prefix = preprocess_prefix
		self.train_split = train_split
		self.max_vocab_len = max_vocab_len
		self.max_feature_value = max_vocab_len
		self.max_sequence_len = max_sequence_len

		self.ignore_class = ignore_class
		self.classes_path = classes_path
		if classes_path is not None and os.path.exists(classes_path):
			with open(classes_path) as fp:
				self.classes = [line.strip() for line in fp]
		else:
			self.classes_path = classes_path
			self.classes = None

		self.word_to_idx = None
		self.idx_to_word = None

		self.ngram_to_feat = None

		self.train_text = None
		self.val_text = None
		self.train_label = None
		self.val_label = None

		self.train_size = None
		self.val_size = None
		
		if os.path.exists(stopwords_path):
			self._load_stopwords()

		if os.path.exists(vocab_path):
			self._load_vocab()
		else:
			self.load_data()
			
	def _load_stopwords(self):
		with open(self.stopwords_path) as fp:
			self.stopwords = [line.strip() for line in fp]

	def _load_vocab(self):
		vocab_list = []
		if os.path.exists(self.vocab_path):
			with open(self.vocab_path) as fp:
				vocab_list = [line.strip() for line in fp]

		self.word_to_idx = {word:index + 1 for index, word in enumerate(vocab_list)}	# 0 for pad
		self.idx_to_word = {index + 1: word for index, word in enumerate(vocab_list)}
		return vocab_list

	def create_vocab(self, 
			 text_list, 
			 max_count = None, 
			 save_flag = True):
		counter = Counter([word for text in text_list for word in text.split()])    
		top_words_with_counts = counter.most_common(max_count)
		vocab = [word for word, _ in top_words_with_counts][:self.max_vocab_len - 1]
		vocab += ['UNK']

		self.word_to_idx = {word:index + 1 for index, word in enumerate(vocab)}		# 0 for pad
		self.idx_to_word = {index + 1: word for index, word in enumerate(vocab)}

		if save_flag:
			with open(self.vocab_path, "w") as fp:
				fp.write("\n".join(vocab))

		return vocab

	def preprocess_data(self,
			    data_path, 
			    mode = 'train',
			    savefile_prefix = "preprocessed_data",
			    save_flag = False):
		count =  0 
		text_list = []
		label_list = []
		for label in os.listdir(data_path):
			for filename in os.listdir(os.path.join(data_path, label)):
				with open(os.path.join(data_path, label, filename)) as fp:
					text = fp.readlines()
				filtered_text = self.filter_text(text)

				text_list.append(filtered_text)
				label_list.append(label)
				count += 1
		print("Total {} data read!".format(count))

		data = list(zip(text_list, label_list))
		
		df = pd.DataFrame(data = data, columns=['text', 'label'])
		if save_flag:
			save_path = "{}_{}.csv".format(savefile_prefix, mode)
			df.to_csv(save_path, index = False, header = True)

		return df

	def tokenize(self, text):
		return [word for word in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", text) \
			if word != '' and word != ' ' and word != '\n']

	def filter_text(self, text):
		text = [word for word in self.tokenize(text.lower()) if word not in self.stopwords]
		text = ' '.join(text)

		return text

	def create_ngrams(self, data, ngram_size = 2):
		return set(zip(*[data[index:] for index in range(ngram_size)]))

	def append_ngrams(self, data, ngram_to_feat, ngram_range = 2, select = 'all'):
		"""Adds ngram feature values to corresponding unigram feature list representing text"""
		result = [] 
		for feature_list in data:
			out_list = feature_list[:]
			for ngram_size in range(2, ngram_range + 1):
				ngrams_per_size = zip(*[feature_list[index:] for index in range(ngram_size)])
				valid_features = [ngram_to_feat[ngram] for ngram in ngrams_per_size \
						 if ngram in ngram_to_feat]
				out_list += valid_features
			if select == 'random':
				out_len = int(np.random.uniform(0.5, 1.) * len(out_list))
				out_list = list(np.random.choice(out_list, out_len, replace = False))
			result.append(out_list)     
		return result

	def enable_ngrams(self, data, ngram_range = 2, max_feature_value = None, mode = 'train', select = 'all'):
		if ngram_range < 2:
			return data

		if mode == 'train' and self.ngram_to_feat is None:
			select = 'all'
			ngram_set = set()
			for feature_list in data:
				for ngram_size in range(2, ngram_range + 1):
					temp = self.create_ngrams(feature_list, ngram_size)
					ngram_set.update(temp)

			if max_feature_value is None:
				if self.max_feature_value is None:
					max_feature_value = max([max(feature_list) for feature_list in data])
				else:
					max_feature_value = self.max_feature_value

			start_index = max_feature_value + 1
			self.ngram_to_feat = {ngram : index + start_index for index, ngram in enumerate(ngram_set)}
			self.max_feature_value = np.max(list(self.ngram_to_feat.values())) + 1

		data = self.append_ngrams(data, self.ngram_to_feat, ngram_range, select = select)
		return data

	
	def encode_text(self, text_data):
		text_data = [[self.word_to_idx[word] for word in text.split()] for text in text_data]
		return text_data

	def decode_text(self, text_data):
		text_data = [[self.idx_to_word[idx] for idx in text] for text in text_data]
		text_data = ["".join(text_list) for text_list in  text_data]  
		return text_data

	def encode_label(self, label_data):
		label_data = [[1 if tag in label_per_text else 0 for tag in self.classes] \
			      for label_per_text in label_data]

		label_data = [[1] + label_per_text[1:] if sum(label_per_text) == 0 else label_per_text \
			      for label_per_text in label_data]
		return label_data

	def decode_label(self, label_data):
		label_data = [[1 if prob > 0.5 else 0 for prob in prob_per_text] \
			      for prob_per_text in label_data]
		label_data = [[tag for idx, tag in enumerate(self.classes) \
			      if label_per_text[idx] == 1] for label_per_text in label_data]
		label_data = [label_per_text[1:] if len(label_per_text) > 1 else label_per_text \
			      for label_per_text in label_data] 					# Discount ignores if other labels
		return label_data

	def load_data(self, 
		      train_data_path = None, 
		      val_data_path = None, 
		      savefile_prefix = "preprocessed_data"):
		pre_train_data_path = "{}_{}.csv".format(savefile_prefix, 'train')
		pre_val_data_path = "{}_{}.csv".format(savefile_prefix, 'val')

		train_data_path = self.data_path if train_data_path is None else train_data_path

		data = None
		if not os.path.exists(os.path.join(self.data_path, pre_train_data_path)):
			if os.path.exists(train_data_path):
				data = self.preprocess_data(train_data_path, save_flag = True)
		else:
			data = pd.read_csv(os.path.join(self.data_path, pre_train_data_path), header = 0)

		if data is None:
			return
	
		train_text, train_label = data["text"], data["label"]
        
		data = None
		if not os.path.exists(os.path.join(self.data_path, pre_val_data_path)):
			if val_data_path is not None and os.path.exists(val_data_path):
				data = self.preprocess_data(val_data_path, save_flag = True)
		else:
			data = pd.read_csv(os.path.join(self.data_path, pre_train_data_path), header = 0)

		if not os.path.exists(self.vocab_path):      
			self.create_vocab(train_text, self.max_vocab_len)

		if self.classes is None:
			self.classes = [self.ignore_class] + list(set([label_list.split(',')[0] for label_list in train_label]))
			if self.classes_path is not None:
				with open(self.classes_path, 'w') as fp:
					for class_name in self.classes:
						fp.write(class_name + '\n')

		if data is None:
			train_count = int(len(train_text) * self.train_split)
			train_text, train_label, val_text, val_label = train_text[:train_count], train_label[:train_count], \
								       train_text[train_count:], train_label[train_count:]
		else:
			val_text, val_label = data["text"], data["label"]

		train_text = self.encode_text(train_text)
		train_label = self.encode_label(train_label)

		val_text = self.encode_text(val_text)
		val_label = self.encode_label(val_label)

		train_text = self.enable_ngrams(train_text, ngram_range = 1, mode = 'train')
		val_text = self.enable_ngrams(val_text, ngram_range = 1, mode = 'val')

		print('Average train sequence length: {}'.format(np.mean(list(map(len, train_text)), dtype = int)))
		print('Average val sequence length: {}'.format(np.mean(list(map(len, val_text)), dtype = int)))

		self.train_size = len(train_text)
		self.val_size = len(val_text)

		self.train_text, self.train_label, self.val_text, self.val_label = train_text, train_label, val_text, val_label

	def pad(self, data, fixed = True):
		if fixed == True and self.max_sequence_len is not None:
			maxlen = self.max_sequence_len
		else:
			maxlen = max([len(features) for features in data])
        
		paddings = [[0] * (maxlen - len(features)) for features in data]      
		data = [feat_list[:maxlen] + padding for feat_list, padding in zip(data, paddings)]
		return data

	def generate(self, batch_size = 32, mode = 'train'):
		if mode == 'train':
			text = self.train_text
			label = self.train_label
		else:
			text = self.val_text
			label = self.val_label
		
            
		batch_index = 0
		while True:
			if (batch_index + 1) * batch_size >= len(text):
				batch_index = 0
				if mode == 'train':              
					idx_list = list(range(len(text)))
					np.random.shuffle(idx_list)
					text = [text[idx] for idx in idx_list]
					label = [label[idx] for idx in idx_list]

			x = []
			y = []
			for index in range(batch_size):
				x.append(text[batch_index * batch_size + index])
				y.append(label[batch_index * batch_size + index])
			x = self.pad(x)
		
			batch_index += 1
			yield np.array(x), np.array(y)
