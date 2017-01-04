# NapEasy
NapEasy is an automated PDF highlighting tool designed to support faster curation of case studies and reviews on Parkinson's and Alzheimer's disease. Now, it has been deployed as a web server at [http://napeasy.org](http://napeasy.org), which allows the user to highlight selected papers from the PubMed Central (PMC) Open Access dataset. Visit the [website](http://napeasy.org) for detail. You can also check the following instructions for getting the tool running in your own environment.

## Prerequisite 
Follow this [wiki page](https://github.com/Honghan/NapEasy/wiki) to install required libraries  

## Usages
The functionalities are separated into three parts so that users might pick up the needed functionality or combination of functionailites according to their own situation.

### PDF Preprocessing - PDF to XML Converting and Highlight Extraction
```
python pdf_processor.py INPUT_FOLDER_PATH OUTPUT_FOLDER_PATH
```
The above command takes as input a path that contains all the PDF files to be processed, and generates XML files (converted from PDFs) and text files(highlights from PDFs) for all input PDF files.

### Annotating Papers, and Merging Annotations and Highlights
```
python article_ann.py XML_FOLDER_PATH
```
This command annotates all papers (xml format) using NCBO Annotator, and merge the annotations with extracted highlights (if they appear in the same folder). Note that the ontologies used in the annotation are `FMA, GO, HP, and PATO`. This can be cusotmised by modifying `line 22` of this python script. The process is multithreaded with a default setting of 30 threads. This can be changed at `line 346`.

### Highlighting Papers
```
python ht_experiments.py ht ANN_FOLDER_PATH OUTPUT_FOLDER_PATH
```
This command generates highlights for all papers in the `ANN_FOLDER_PATH`. Note: the folder should contain the merged annotations from the previous step. There will be two steps:
 - step 1: extracting language patterns for all sentences and generate spatial regions for each paper; all these intermediate results will be saved as a json file for each paper with a suffix `_score.json`.
 - step 2: automated highlighting; all results will be saved into one json file in the output folder with a name of `highlight-results.json`.

## Evaluating performances using our corpus
```
python ht_experiments.py
```
this command will output the detailed performance outputs on our corpus. For each set of papers, it includes the result on each paper ([view](/results/macro_results.tsv)) and also the miro-average result. The corpus detail is as follows.

For the purpose of this study, we investigated 235 full text papers that have been manually curated and highlighted by a senior curator. The highlights within each paper are used to create manual summaries that report on the correlation between brain anatomy and neuropsychometric tests (i.e. brain function). The set of papers was divided into two sets, one for deriving language patterns and scoring thresholds (183 papers with at least one highlight out of total 205 papers) [view detail](results/configuration_set_stats.tsv) and one for testing the suggested methodology (18 papers) [view detail](results/evaluation_set_stats.tsv). The papers contained in the collection for the derivation of language patterns included 86 different journals, while the collection for evaluation purposes included 17 different journals (12 journals that were contained in both paper collections). 

We note here that the highlighting is more difficult than well-studied text summary tasks because it is not possible to pre-determine the number of sentences that need to be highlighted in the paper due to the large variation of sentences highlighted in PDFs. Please check the following plot that shows the distribution of highlights vs. sentences in our corpus.
![Alt](/results/sentences_highlights.png "#highlights vs. #sentences")
