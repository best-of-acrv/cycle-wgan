import argparse

class Arguments():

	def __init__(self):
		self.parser = argparse.ArgumentParser(description="Multi-modal cycle-consistent GZSL")
		self.parser._action_groups.pop()
		required = self.parser.add_argument_group('required arguments')
		optional = self.parser.add_argument_group('optional arguments')

		required.add_argument("--config", help="JSON file containing experiment configuration", required=True)
		optional.add_argument("--gpu", help="GPU index", type=int, dest='gpu_id', default=0)
		optional.add_argument("--cpu", help="CPU mode", type=bool, default=False)

		#If workdir is not provided, a unique directory will be created by the program
		#If workdir is provided, model checkpoints will be loaded if they exist 
		optional.add_argument("--work-dir", help="Base directory for reading/writing - unique directory created by program if not provided", type=str, dest='work_dir', default=None)

		#GAN arguments
		optional.add_argument("--train-gan", help="Include this flag to train the GAN", dest='train_GAN', action='store_true')
		
		#Fake dataset generation arguments
		optional.add_argument("--gen-fake", help="Include this flag to generate fake dataset", dest='generate', action='store_true')
		optional.add_argument("--domain", nargs='+', help="Generate 'unseen', 'seen' or 'unseen' 'seen'", type=str, default=['unseen'])
		optional.add_argument("--num-features", nargs='+', help="Number of features to generate e.g. 200, or 200 150", type=int, dest='num_features', default=[200])

		#Fake data augmentation arguements
		optional.add_argument("--aug-file", help="File to read/write fake dataset. Default location in work-dir used if not provided", type=str, dest='aug_file', default=None)
		optional.add_argument("--aug-op", help="Data augmentation option 'merge' (real+aug), 'replace' (aug only) or 'none' (no augmentation)", type=str, dest='aug_op', default="merge")

		#GZSL classifier arguments
		optional.add_argument("--train-cls", help="Include this flag to train GZSL classifier regressor", dest='train_GZSL', action='store_true')
		optional.add_argument("--test-cls", help="Include this flag to test a trained GZSL classifier regressor", dest='test_GZSL', action='store_true')

	def parse(self):
		self.args = self.parser.parse_args()
		return self.args	

