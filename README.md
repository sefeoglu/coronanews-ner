# Fine-Grained Named Entities for Corona News
The work in this repository had been presented at the SWAT4HCLS conference 2023, Basel, Switzerlands on February 15, 2023.

The presentation is available [here](https://repository.publisso.de/resource/frl%3A6440380) and paper will be available soon at the inproceedings of the conference.

The sample sentence:
````python

sample_sentence_en = '''Lauterbach: Omicron is not suitable as a vaccine substitute Federal Health Minister Karl Lauterbach refers to a study from South Africa on Twitter, according to which an Omicron infection in unvaccinated people hardly protects against a disease with the delta variant .'''

model = SequenceTagger.load(model_path)

sentence = Sentence(sample_sentence_en)

model.predict(sentence)

for entity in sentence.get_spans("ner"):
  print(entity)
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

````python
sentence = """ How many people want to use the app?
2463.74, According to the ARD Germany trend from June, 42 percent of those surveyed would use such a warning app on their own smartphone, while 39 percent would not."""
````
Output:
```
Span[2:3]: "people" → GROUP (1.0)
Span[9:10]: "2463.74" → CARDINAL (0.9356)
Span[14:15]: "ARD" → ORG (0.9996)
Span[15:16]: "Germany" → GPE (0.9995)
Span[18:19]: "June" → DATE (0.9871)
Span[20:22]: "42 percent" → PERCENT (0.9954)
Span[37:39]: "39 percent" → PERCENT (1.0)
```


## Demo

The usage is available [here](https://github.com/sefeoglu/coronanews-ner/blob/master/src/viz/A_NER_Model_for_Corona__News.ipynb)

## Note:
The NER model is [here](https://drive.google.com/file/d/1R6WVbynZK81J_aeBkRyTu2koHTf4hTpF/view?usp=sharing)
