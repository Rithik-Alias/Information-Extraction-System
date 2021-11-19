# Information-Extraction-System

Creating a information extraction system include 4 phases
* Annotation of Documents
* Creation of a custom NER model for the data.
* Creation of Relation Extraction model for the extracted entities.
* Creation of a neo4j database using these relations and entities.

## Annotation of Documents

You can use UBIAI tool to do annotation of pdf documents. It has very good interface for annotation. But its not open source!!!
The link to UBIAI : UBIAI
Or else you can go for label-studio The details about installation of label studio is available in its GitHub repository. label-studio
Export your annotated file in .conll format.

And for RE model, the data can be annotated in the UBIAI tool. And export the data in json format.

## Creation of a custom NER model

I used Spacy transformers using BERT here to create the model.

!python -m spacy convert Dev.conll ./ -t json -n 3 -c iob
!python -m spacy convert Train.conll ./ -t json -n 3 -c iob
If you are using UBIAI tool replace the .conll files with .tsv files that you get from UBIAI tool.

!python -m spacy init fill-config base_config.cfg config.cfg The base_config file specified in this line is available from (https://spacy.io/usage/training?ref=hackernoon.com) 

After the training the model will be saved in the directory NER/model-best/

You can test the model and the python notebook for training the model is available in Test folder.

Unfortunately from spacy 3.0 onwards, there is no option to get the confidence score of extracted entities.

You can get the validation score while training in meta.json file inside the model folder.

Refer to the following blog for creation of the NER model.\
https://towardsdatascience.com/how-to-fine-tune-bert-transformer-with-spacy-3-6a90bfe57647

## Creation of Relation Extraction model

I used Spacy transformers using BERT here to create the model.
The binary_converter.py file that I used in the code `!python3 binary_converter.py` is available in the dependencies folder.

Also, run `!spacy project run evaluate ` to do validation of the model.


Run the following code to test the model.
Also, you can change the threshold value for extracted entities here,\
`if max(rel_dict.values()) >=0.5 :`
    
    import random
    import typer
    from pathlib import Path
    import spacy
    from spacy.tokens import DocBin, Doc
    from spacy.training.example import Example
    from rel_pipe import make_relation_extractor, score_relations
    from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors 
    nlp2 = spacy.load("/content/drive/MyDrive/NER_RE_New/RE/rel_component/training/model-best")
    nlp2.add_pipe('sentencizer')  ## Added to avoid error [E030] Sentence boundaries unset.

    def relation(text):
      for doc in nlp.pipe(text, disable=["tagger", "parser"]):
          #print([(ent.start,ent.end,ent.text, ent.label_) for ent in doc.ents])
          entity = [(e.start,e.end, e.text, e.label_) for e in doc.ents]
          for name, proc in nlp2.pipeline:
            doc = proc(doc)
          # Here, we split the paragraph into sentences and apply the relation extraction for each pair of entities found in each sentence.
          rel=[]
          for value, rel_dict in doc._.rel.items():
            for sent in doc.sents:
              for e in sent.ents:
                  for b in sent.ents:             
                    if (e.start == value[0] and b.start == value[1]):
                      if max(rel_dict.values()) >=0.5 :
                        if (e.start < b.start):
                          #print(rel_dict)
                          print(f" Entities: {e.start,b.start,e.text, b.text} --> predicted relation: {rel_dict}")  
                          print(f"Relations: {e.start,b.start,e.text, b.text,max(rel_dict, key=rel_dict.get)}")
                          #print(f"Relations: {e.text, b.text,list(rel_dict.keys())[0]}")
                          temp = (e.text, b.text,max(rel_dict, key=rel_dict.get))
                          rel.append(temp)
      print('Entities:')
      print(entity)
      print('Relation:')
      print(rel)`



Refer to the following blog for creation of the RE model.\
https://towardsdatascience.com/how-to-train-a-joint-entities-and-relation-extraction-classifier-using-bert-transformer-with-spacy-49eb08d91b5c

You can also refer to the following colab notebook file link.\
https://colab.research.google.com/drive/1vkuTHE_HTJAiCpAfg8JDMfDTCuOEkVIw?usp=sharing

## Creation of a neo4j database using these relations and entities.

