import os
import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from data_loader import DataLoader
from text_classifier import Classifier

def train(batch_size = 32, save_path = 'chk_classifier.hdf5', resume_flag = False):

	checkpointer = ModelCheckpoint(monitor = 'val_acc', 
				       filepath = save_path, 
				       verbose = 1, 
				       save_best_only = True, 
				       save_weights_only = True)
	early_stopper = EarlyStopping(monitor = 'val_loss', 
				      min_delta = 0.001, 
				      patience = 20)
	lr_reducer = ReduceLROnPlateau(monitor = 'val_loss',
				       factor = 0.8,
				       verbose = 1,
				       patience = 4,
				       min_lr = 2E-6)

	data_loader = DataLoader(data_path = 'data')
	data_loader.load_data()

	model = Classifier(data_loader.max_feature_value,
			   data_loader.max_sequence_len,
			   num_classes = len(data_loader.classes))

	classifier = model()
	classifier.summary()
	train_steps = data_loader.train_size // batch_size
	val_steps = data_loader.val_size // batch_size

	if resume_flag and os.path.exists(save_path):
		classifier.load_weights(save_path)

	classifier.fit_generator(generator = data_loader.generate(mode = 'train'),
			    steps_per_epoch = train_steps,
			    epochs = 350,
			    verbose = 1,
			    validation_data = data_loader.generate(mode = 'val'),
			    validation_steps = val_steps,
			    callbacks = [checkpointer, early_stopper, lr_reducer])

def test(batch_size = 1, save_path = 'chk_classifier.hdf5'):
	data_loader = DataLoader(data_path = 'data')
	data_loader.load_data()

	model = Classifier(data_loader.max_feature_value,
			   data_loader.max_sequence_len,
			   num_classes = len(data_loader.classes))

	classifier = model()
	classifier.load_weights(save_path)

	generator = data_loader.generate(batch_size = batch_size, mode = 'val')
	x, y = next(generator)
	
	pred = classifier.predict(x)
	y_in = data_loader.decode_label(y)
	y_out = data_loader.decode_label(pred)
	
	[print('Ground Truth: {} Prediction: {}'.format(y1, y2)) for y1, y2 in zip(y_in, y_out)]
	metrics = classifier.evaluate(x, y, verbose = 1)
	[print('{}: {}\n'.format(x, y)) for x, y in zip(classifier.metrics_names, metrics)]
	print(metrics)

if __name__  == '__main__':
	parser = argparse.ArgumentParser(description = 'SortNet Parameters')
	
	parser.add_argument('-m',
			    '--exec_mode',
			    default = 'train',
			    help = 'Execution mode',
			    choices = ['train', 'test'])
	parser.add_argument('-b',
			    '--batch_size',
			    default = 32)
	parser.add_argument('-s',
			    '--save_path',
			    default = 'chk_classifier.hdf5')

	arguments = parser.parse_args()
	mode = arguments.exec_mode
	batch_size = arguments.batch_size
	save_path = arguments.save_path

	if mode == 'train':
		train(batch_size, save_path, resume_flag = True)
	else:
		test(batch_size, save_path)
	
