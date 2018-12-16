import re
import pandas as pd

def preprocess_data(data_path,
		    feature_header = 'Description',
		    label_header = 'Label'
		    mode = 'train',
		    savefile_prefix = "preprocessed_data",
		    save_flag = True):

	xls = pd.ExcelFile(data_path)
	df = xls.parse(xls.sheet_names[0])
	df = df[pd.notnull(df[label_header])]
	df.fillna('', inplace=True)  

	description = df[feature_header]
	label = df[label_header]
	print("Data Load Complete!")
    
	text = [description for description in list(map(filter_text, description))]

	print("Total {} data read!".format(len(label)))

	data = list(zip(text, label))
	
	df = pd.DataFrame(data = data, columns=['text', 'label'])
	if save_flag:
		save_path = "{}_{}.csv".format(savefile_prefix, mode)
		df.to_csv(save_path, index = False, header = True)

	return df

def tokenize(text):
		return [word for word in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", text) \
			if word != '' and word != ' ' and word != '\n']

def filter_text(input_text):
	english_words = set(words.words('en'))
	stop_words = set(stopwords.words('english'))

	def replace_name(matchobj):
		word = matchobj.group(0)
		replacement = '' if word.lower() not in english_words else word
		return replacement

	text = re.sub(r'&nbsp;', ' ', input_text)		# Substitute whitespaces
	text = re.sub(r'<.*?>|&.*;|[^\s\w]+', ' ', text)	# Remove html markups
	text = re.sub(r'\w+', replace_name, text)		# Remove non-dictionary words
	text = re.sub(r'\b\w\b', '', text)			# Remove single character words
	text = re.sub(r' +', ' ', text)				# Squeeze multiple white spaces
	text = [word for word in tokenize(text.lower()) \
		if word not in stop_words]			# Remove stopwords
	text = ' '.join(text)   
	return text

if __name__ == '__main__':
	file_path = 'data/data.xlsx'
	preprocess_data(file_path)


