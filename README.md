# Topic-Modelling
Topic Modeling of Covid-19 Tweets Using BERTopic

In this topic modelling analysis, we explore the application of topic modelling techniques to uncover latent themes within a corpus of tweets related to COVID-19. Social media platforms, especially Twitter, have become essential for understanding public sentiment, identifying trending topics, and gaining insights into various social and political events. By analysing these tweets, we aim to extract meaningful topics to help understand the broader narrative and opinions expressed during the pandemic.
We will utilise the BERTopic model that leverages transformer-based language models for generating document embeddings, clustering these embeddings, and extracting coherent topic representations through a class-based variation of TF-IDF. This model addresses limitations in traditional methods like Latent Dirichlet Allocation (LDA) and Non-Negative Matrix Factorization (NMF) by incorporating semantic relationships and contextual word representations.

The dataset of COVID-19-related tweets from Kaggle was selected. A random subset of 5000 tweets was taken from the data set. 
https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification?select=Corona_NLP_train.csv


Topic Modelling
BERTopic is a sophisticated topic modelling technique developed by Maarten Grootendorst. It leverages BERT embeddings and a class-based TF-IDF approach to create dense, coherent clusters, enabling the automatic extraction of meaningful topics from large volumes of unstructured text. This overview details the key steps and parameters involved in implementing BERTopic.

1. Embed Documents
Firstly, we need to get the embeddings for all the documents. Embeddings are the vector representation of the documents. BERTopic uses the English version of the sentence_transformers ("all-MiniLM-L6-v2”) by default to get document embeddings. BERTopic supports the pre-trained models from other Python packages, such as hugging face and flair. We use the default BERTopic embedding here, and the language was set to english. Also, we set the calculate_probabilities parameter to True.

2. Dimensionality Reduction
BERTopic employs UMAP (Uniform Manifold Approximation and Projection) to handle high-dimensional data for dimensionality reduction. UMAP preserves local and global data structures, which is essential for creating clusters of semantically similar documents. Although UMAP is the default, other techniques like PCA can also be used based on the use case. We employed UMAP with default parameters for this context.

3. Cluster Documents
After reducing the embeddings' dimensions, BERTopic clusters the data using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise). HDBSCAN can identify clusters of varying shapes and densities and detect outliers, ensuring that documents are not forced into inappropriate clusters, thus enhancing topic representation quality. We also employed the HDBSCAN with default parameters.

4. Topic Representation
BERTopic combines all documents in a cluster into one document to create topic representations using a modified TF-IDF approach called class-based TF-IDF (c-TF-IDF). This approach calculates the importance of words within each cluster, providing coherent topic descriptions. By comparing word importance between clusters instead of individual documents, c-TF-IDF offers a better representation of topics, highlighting key themes more effectively. The default BERTopic c-TF-IDF was used for our analysis as well.
