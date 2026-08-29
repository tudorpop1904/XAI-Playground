"""
core/detectors/kmc.py

Implements k-Means Clustering (KMC) detection of AI deepfakes.

WHAT IS KMC?
------------

k-Means Clustering (which we will nonstandardly abbreviate to KMC) is an unsupervised learning
algorithm which partitions a set of data points into k clusters (one for each class).

Unlike k-NN, KMC doesn't possess any knowledge of the classes of the training images during training.
KMC acquires knowledge of the classes of the training images by clustering them, then assigning the
label of the majority of the data points in each cluster to the cluster.

During inference, KMC assigns each input image to the cluster whose centroid is closest to the input
image's feature vector. The distance metric used is cosine similarity (see core/detectors/knn.py).

How it differs from the other three classifiers:

  ┌──────────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
  │     Property     │     CNN      │     ViT      │     k-NN     │    KMC       │
  ├──────────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
  │ Paradigm         │ Supervised   │ Supervised   │ Unsupervised │ Unsupervised |
  │ Type             │ Parametric   │ Parametric   │Non-parametric│Non-parametric│
  │ Learns weights?  │ Yes (~470K)  │ Yes (~5.5M)  │ No (0)       │ No (0)       │
  │ Training         │ Backprop     │ Backprop     │ Store vectors│ Store vectors│
  │ Inference speed  │ O(1) forward │ O(1) forward │ O(N) search  │ O(N) search  │
  │ XAI method       │ Grad-CAM/VGS │ Attn rollout │ Show k nbrs  │ Show cluster │
  │ Decision boundary│ Smooth       │ Smooth       │ Jagged       │ Jagged       │
  └──────────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

WHY INCLUDE KMC?
----------------
1. We introduce k-means clustering as a fourth pillar to our research basis, since it has the
benefit of being a clustering algorithm that is trained on the same feature space as k-NN,
therefore allowing us to directly compare the two algorithms.

2. While k-means clustering has no learnable parameters, it is still a "parametric" algorithm in
the sense that the cluster centroids are learned from the training data.

The KMC algorithm works as follows:

1. Initialize the centroids of the k clusters to be the feature vectors of k random images
   from the training set.
2. Assign each image to the cluster whose centroid is closest to the image's feature vector.
3. Recalculate the centroids of each cluster to be the mean of the feature vectors of the
   images assigned to that cluster.
4. Repeat steps 2 and 3 until the centroids no longer change significantly.

TRICK
-----
We employ the same trick as for the k-NN classifier, i.e., converting each image into a latent
vector representation, for better memory management, computational resources, and ease of comparison.



"""