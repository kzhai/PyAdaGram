PyAdaGram
==========

PyAdaGram is an online Adaptor Grammar model package, developed by the Cloud Computing Research Team in [University of Maryland, College Park] (http://www.umd.edu).

Please download the latest version from our [GitHub repository](https://github.com/kzhai/PyAdaGram).

Please send any bugs of problems to Ke Zhai (kzhai@umd.edu).

Install and Build
----------

This package depends on many external python libraries, such as numpy, scipy and nltk.

Launch and Execute
----------

Assume the PyAdaGram package is downloaded under directory ```$PROJECT_SPACE/src/```, i.e., 

	$PROJECT_SPACE/src/PyAdaGram

To prepare the example dataset,

	tar zxvf brent.tar.gz

To launch PyAdaGram, first redirect to the parent directory of PyAdaGram source code,

	cd $PROJECT_SPACE/src/

and run the following command on example dataset,

	python -m PyAdaGram.launch_train --input_directory=./PyAdaGram/brent/ --output_directory=./PyAdaGram/ --grammar_file=./PyAdaGram/brent/grammar.unigram --number_of_documents=9790 --batch_size=10

The generic argument to run PyAdaGram is

	python -m PyAdaGram.launch_train --input_directory=$INPUT_DIRECTORY/$CORPUS_NAME --output_directory=$OUTPUT_DIRECTORY --grammar_file=$GRAMMAR_FILE --number_of_documents=$NUMBER_OF_DOCUMENTS --batch_size=$BATCH_SIZE

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any cirsumstances, you may also get help information and usage hints by running the following command

	python -m PyAdaGram.launch_train --help
