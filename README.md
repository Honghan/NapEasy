# Automated PDF highlighting tool to support faster curation of case studies and reviews on Parkinson's and Alzheimer's disease
## usages
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
