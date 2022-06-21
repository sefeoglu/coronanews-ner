# A NER model for Corona News

## Data Preparation

### * 1.) Seed Selection
The seed entities were selected from wikidata and cord-ner dataset[1] for the following categories:

IMMUNE RESPONSE, CORONAVIRUS, DISEASE OR SYNDROME, SIGN OR SYMPTOM

### * 2.) Annotation of the Training Text with Domain-Specific Seeds

### * 3.) Annotation of the Training Text with Flair and validation of them with Wikidata

### * 4.) Test Data Preparation with Domain Experts
Two domain experts annotated 2000 sentences from the text data and its Fleiss kappa is 0. for the following categories: EVENT, PRODUCT, IMMUNE RESPONSE, CORONAVIRUS, DISEASE OR SYNDROME, SIGN OR SYMPTOM

## Model Training

## Demo

The usage is available [here](https://github.com/sefeoglu/coronanews-ner/blob/master/src/viz/A_NER_Model_for_Corona__News.ipynb)

=======
## References:
[1] Wang, X., Song, X., Li, B., Guan, Y., Han, J.: Comprehensive named entity recognition on cord-19 with distant or weak supervision (2020)

