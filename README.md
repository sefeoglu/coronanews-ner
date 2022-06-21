# A NER model for Corona News





The sample sentence:
````python

sample_sentence_en = '''Lauterbach: Omikron is not suitable as a vaccine substitute Federal Health Minister Karl Lauterbach refers to a study from South Africa on Twitter, according to which an Omikron infection in unvaccinated people hardly protects against a disease with the delta variant .'''

model = SequenceTagger.load(model_path)

sentence = Sentence(sample_sentence_en)

model.predict(sentence)

for span in sentence.get_spans("ner"):
  print(span)
````
Output:
````
Span[0:1]: "Lauterbach" → PERSON (0.9999)
Span[2:3]: "Omicron" → CORONAVIRUS (1.0)
Span[8:9]: "vaccine" → PRODUCT (0.9746)
Span[13:15]: "Karl Lauterbach" → PERSON (0.9997)
Span[20:22]: "South Africa" → GPE (0.9675)
Span[23:24]: "Twitter" → ORG (0.9986)
Span[29:31]: "Omicron infection" → DISEASE_OR_SYNDROME (0.9965)
Span[32:34]: "unvaccinated people" → GROUP (0.998)
Span[38:39]: "disease" → DISEASE_OR_SYNDROME (0.851)
Span[41:43]: "delta variant" → CORONAVIRUS (0.9608)
````



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

