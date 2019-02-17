# Semantic Text Segmentation with Embeddings

Given a text document *d* and a desired number of segments *k*, 
this repo shows how to segment the document into *k* semantically homoginuous segments. 

The general approach is as follows:
1. Convert the words in *d* into embedding vectors using the GloVe model.
2. For all word sequences *s* in *d*: the average meaning (centroid) of *s* is represented by taking the average embeddings of all words in *s*.
3. The error in the centroid calculation for a sequence *s* is calculated as the average cosine distance between the centroid and all words in *s*.
4. The segmentation is done using the greedy heuristic by iteratively choosing the best split point *p*.


The class text_segmentation_class in __[text_segmetnation.py](https://github.com/ReemHal/Semantic-Text-Segmentation-with-Embeddings/blob/master/text_segmentation.py)__ contains funtions to convert the document
words in GloVe embeddings and choose the splitting points. The notebook __[semantc_text_segmentation_example.ipynb](https://github.com/ReemHal/Semantic-Text-Segmentation-with-Embeddings/blob/master/semantc_text_segmentation_example.ipynb)__
demonstrates how to use the class.

<a name="tech"><a/>
## Technologies

This project was developed using:

  - python=3.5
  - gensim==3.7.1
  - matplotlib==2.2.3
  - mpld3==0.3
  - nltk==3.4
  - numpy==1.14.5
  - pandas==0.23.4
  - parso==0.3.1
  - Pillow==5.2.0
  - scikit-learn==0.19.2
  - scipy==1.1.0

