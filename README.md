# Topic Modeling of Covid-19 Tweets Using BERTopic

## Introduction
In this topic modelling analysis, we explore the application of topic modelling techniques to uncover latent themes within a corpus of tweets related to COVID-19. Social media platforms, especially Twitter, have become essential for understanding public sentiment, identifying trending topics, and gaining insights into various social and political events. By analysing these tweets, we aim to extract meaningful topics to help understand the broader narrative and opinions expressed during the pandemic.
We will utilise the BERTopic model that leverages transformer-based language models for generating document embeddings, clustering these embeddings, and extracting coherent topic representations through a class-based variation of TF-IDF. This model addresses limitations in traditional methods like Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) by incorporating semantic relationships and contextual word representations [3].


The dataset of COVID-19-related tweets from Kaggle was selected. A random subset of 5000 tweets was taken from the data set. 
https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv

## Data Collection and preprocessing
For this analysis, we have selected a dataset of COVID-19-related tweets from Kaggle [1]. A random subset of 5000 tweets was taken from the data set. The following data preprocessing steps were taken.
The following items were removed:
•	Mentions
•	Hashtags
•	URLs
•	Digits
•	Emojis
•	Non-English words
•	Stop Words
•	Converted all words to lowercase to avoid repetitions.
•	Used lemmatisation to maintain grammatical context and meaningful word forms.
Lemmatisation was used to keep tokens meaningful while avoiding grammatical errors. As per the documentation, removing stop words was not essential for BERTopic [2]. However, I could see a few stop words appearing in some of the topics. Hence, I tried stopwords removal and fit the model again. The resulting topics were more meaningful than earlier.


## Topic Modelling
BERTopic is a sophisticated topic modelling technique developed by Maarten Grootendorst. It leverages BERT embeddings and a class-based TF-IDF approach to create dense, coherent clusters, enabling the automatic extraction of meaningful topics from large volumes of unstructured text. This overview details the key steps and parameters involved in implementing BERTopic.

<img width="481" height="96" alt="image" src="https://github.com/user-attachments/assets/eaac45f1-c186-4728-93b7-febb0bb1f8f7" />


### 1. Embed Documents
Firstly, we need to get the embeddings for all the documents. Embeddings are the vector representation of the documents. BERTopic uses the English version of the sentence_transformers ("all-MiniLM-L6-v2”) by default to get document embeddings [3]. BERTopic supports the pre-trained models from other Python packages, such as hugging face and flair. We use the default BERTopic embedding here, and the language was set to english. Also, we set the calculate_probabilities parameter to True.

### 2. Dimensionality Reduction
BERTopic employs UMAP (Uniform Manifold Approximation and Projection) to handle high-dimensional data for dimensionality reduction. UMAP preserves local and global data structures, which is essential for creating clusters of semantically similar documents. Although UMAP is the default, other techniques like PCA can also be used based on the use case [4]. We employed UMAP with default parameters for this context.

### 3. Cluster Documents
After reducing the embeddings' dimensions, BERTopic clusters the data using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise). HDBSCAN can identify clusters of varying shapes and densities and detect outliers, ensuring that documents are not forced into inappropriate clusters, thus enhancing topic representation quality [4]. We also employed the HDBSCAN with default parameters.

###4. Topic Representation
BERTopic combines all documents in a cluster into one document to create topic representations using a modified TF-IDF approach called class-based TF-IDF (c-TF-IDF). This approach calculates the importance of words within each cluster, providing coherent topic descriptions. By comparing word importance between clusters instead of individual documents, c-TF-IDF offers a better representation of topics, highlighting key themes more effectively [3]. The default BERTopic c-TF-IDF was used for our analysis as well.

## Tools and Libraries:
Google Collab was used as the development platform, and the following tools and libraries were used: bertopoic, gensim, pandas, numpy, nltk, re, string, emoji.

## Evaluation
The topic models' performance was evaluated using the coherence metric. Topic coherence measures the degree of semantic similarity between the words in a topic, indicating how comprehensible the topics are to humans. Specifically, the coherence value (c_v) was used to assess the topic coherence. This is the coherence measure based on the cosine similarity between words in the topic, closely mimicking human judgment.
A coherence score of 0.53 was achieved for the 49 topics. This suggests moderate interpretability, potentially due to data quality issues, inadequate preprocessing, suboptimal topic numbers, model parameter settings, or the choice of dimensionality reduction and embedding techniques. 
To improve coherence, it's essential to experiment with different numbers of topics to find the optimal balance. In this context, I have tried several numbers of topics and calculated the coherence. However, the score did not show a significant improvement. A grid search approach might be useful in finding an optimal number of topics for this scenario. 
Additionally, systematically tuning model parameters using grid search or exploring alternative dimensionality reduction techniques like PCA and trying different embedding or clustering methods can help achieve more meaningful and distinct topics with higher coherence values.

## Results and Discussion
The Intertopic Distance Map visually represents the topics identified in our analysis of COVID-19 tweets. We can see that there are 8 clusters with several sub-clusters inside them. The largest circle represents the more dominant topics (Topic 0 - hand sanitizer and masks, appearing in 459 tweets), suggesting that certain topics are more frequently discussed.

<img width="938" height="938" alt="image" src="https://github.com/user-attachments/assets/e80cf2ff-8882-4684-a218-b1560bc16d6a" />

Figure 1: Intertopic Distance Map


Also, the cluster on the D1 axis talks about topics related to food (mainly topics 5,6 and 9), such as food supply and demand, food donations and panic buying of food (refer to Figure 2) during the pandemic. These topics are positioned close to each other, signifying a higher degree of similarity in their term distributions. Moreover, if we check the similarity scores (refer to Figure 3) of these topics with each other and a topic far away from this topic, such as Topic 0. In that case, we can identify those similar topics (topics that are close to each other) that have a higher similarity than those that are far away. If we investigate further, we can see that in Figure 4, the topics that are similar to one another, which talk about food, are represented using red and are connected to each other. On the other hand, Topic 0 (hand sanitiser and masks) is far away from those topics and connected with related topics like toilet paper and supermarkets.

<img width="958" height="728" alt="image" src="https://github.com/user-attachments/assets/14a853c4-fe22-46d4-b840-e53403be8e83" />

Figure 2: Topic Word Score

Figure 2 highlights key topics related to the COVID-19 pandemic. Major discussions revolve around hygiene products like sanitisers and masks, reflecting concerns about protective measures. Panic buying of toilet paper is another significant theme, as indicated by the focus on terms like "toiletpaper" and "toiletpaperpanic." Food supply issues are also prominent, with keywords pointing to concerns about food stock and demand. Essential workers, such as those in grocery stores and healthcare, are a central topic, highlighting their crucial role during the pandemic. Additionally, there are discussions about scams and fraud, which surged during the crisis. Social distancing practices and changes in consumer behaviour are also notable themes, illustrating the broad impact of the pandemic on daily life and economic activities.


<img width="975" height="975" alt="image" src="https://github.com/user-attachments/assets/e5928892-67f8-490a-a3da-039db98f74f8" />

Figure 3: Similarity Matrix



<img width="975" height="341" alt="image" src="https://github.com/user-attachments/assets/2338fba3-6330-410f-8de7-81cffa119f1d" />

Figure 4: Hierarchical Clustering

The hierarchical clustering visualisation groups the main COVID-19 pandemic topics into distinct clusters, highlighting key concerns. These clusters collectively emphasise the broad impact of the pandemic on economic, social, and public health.


<img width="975" height="609" alt="image" src="https://github.com/user-attachments/assets/c5c8ffa2-e732-4f44-81c5-f8d3abeb69b0" />

Figure 5: Term Score Decline per Topic

In Figure 5, using the elbow method, it seems that 3 words per topic are sufficient to represent most of the topics well. Some topics have variability (4 or 5 words per topic to be representative). Any words that we add after that have seemingly little effect.


<img width="473" height="206" alt="image" src="https://github.com/user-attachments/assets/4de81c92-c85f-4060-b4de-b59e981222a0" />

Figure 6: Topic Probability Distributions

The topic probability distributions show that the first tweet is solely focused on a single topic, namely issues related to workers and employees in grocery stores, highlighting the role of essential workers during the pandemic. In contrast, the second tweet spans several topics, emphasising panic buying of toilet paper.

## Conclusion
This analysis used BERTopic to identify key themes within COVID-19-related tweets, revealing significant topics such as panic buying, food supply issues, hygiene products, essential workers, and social distancing practices. BERTopic's advanced embedding and class-based TF-IDF approach effectively generated coherent topic representations. Despite some limitations, the study shows the potential of sophisticated topic modelling techniques in extracting meaningful insights from social media data. Future research should focus on optimising model parameters for better accuracy and interpretability.

## Limitations
This study has several limitations, including the quality and representativeness of the dataset, which may impact the comprehensiveness of the findings. Despite extensive preprocessing, redundant terms like "covid19" and "coronavirus" may still affect the accuracy of topic modelling. Additionally, the BERTopic model's sensitivity to hyperparameters can lead to varying topic distributions, while dimensionality reduction techniques might not fully capture topic complexity, making interpretation challenging.

## Future Work
Future research should explore different embedding techniques and enhance preprocessing to remove redundant terms. Experimenting with alternative dimensionality reduction methods and optimising hyperparameters could improve topic modelling. Further, systematically optimising hyperparameters for the BERTopic model using grid search could produce more accurate and meaningful topic extraction.

## References 
[1] Miglani, A. "COVID-19 NLP text classification," Kaggle. [Online]. Available: https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv. [Accessed: 02-Jun-2024].
[2] M. Grootendorst, "BERTopic," GitHub. [Online]. Available: https://github.com/MaartenGr/BERTopic. [Accessed: 02-Jun-2024].
[3] M. Grootendorst, "BERTopic: An algorithm for topic modelling," [Online]. Available: https://maartengr.github.io/BERTopic/algorithm/algorithm.html. [Accessed: 02-Jun-2024].
[4] H. Hwang, "Topic modelling with BERT," Towards Data Science, 21-Jun-2020. [Online]. Available: https://towardsdatascience.com/topic-modeling-with-bert-779f7db187e6. [Accessed: 02-Jun-2024].





