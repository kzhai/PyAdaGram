PyAdaGram
==========

PyAdaGram is an online Adaptor Grammar model package, developed by the Cloud Computing Research Team in [University of Maryland, College Park](http://www.umd.edu).
You may find more details about this project on our papaer [Online Adaptor Grammars with Hybrid Inference](http://kzhai.github.io/paper/2014_tacl.pdf) appeared in TACL 2014.

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

	tar zxvf brent-phone.tar.gz

To launch PyAdaGram, first redirect to the directory of PyAdaGram source code,

	cd $PROJECT_SPACE/src/PyAdaGram

and run the following command on example dataset,

```bash
python -m launch_train \
--input_directory=./brent-phone/ \
--output_directory=./ \
--grammar_file=./brent-phone/grammar.unigram \
--number_of_documents=9790 \
--batch_size=10
```

The generic argument to run PyAdaGram is

```bash
python -m launch_train \
--input_directory=$INPUT_DIRECTORY/$CORPUS_NAME \
--output_directory=$OUTPUT_DIRECTORY \
--grammar_file=$GRAMMAR_FILE \
--number_of_documents=$NUMBER_OF_DOCUMENTS \
--batch_size=$BATCH_SIZE
```

You should be able to find the output at directory ```$OUTPUT_DIRECTORY/$CORPUS_NAME```.

Under any circumstances, you may also get help information and usage hints by running the following command

```bash
python -m launch_train --help
```

To launch test script, run the following command

```bash
python -m launch_test \
--input_directory=$DATA_DIRECTORY \
--model_directory=$MODEL_DIRECTORY \
--non_terminal_symbol=$NON_TERMINAL_SYMBOL
```
