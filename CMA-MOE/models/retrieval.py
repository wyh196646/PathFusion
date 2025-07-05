import torch
import torch.nn.functional as F

class Retriever:
    def __init__(
        self,
        similarity = 'l2',
        centering = False
    ):
        """
        Initialize the Retriever.

        Args:
            similarity (str): Similarity metric to use ('cosine' or 'l2').
            centering (bool): Whether to center the embeddings.
            device (str): Device to use ('cpu' or 'cuda:n').
        """
        assert similarity in ['cosine', 'l2'], "Similarity must be 'cosine' or 'l2'."
        self.similarity = similarity
        self.centering = centering

        # Initialize attributes
        self.train_embeddings = None
        self.train_labels = None
        self.center = None

    def fit(self, X_train, y_train):
        """
        Fit the Retriever with training data (keys).

        Args:
            X_train (np.ndarray): Training embeddings (shape: n x d).
            y_train (np.ndarray): Training labels (shape: n).
        """
        self.train_embeddings = X_train
        self.train_labels = y_train
        assert len(self.train_embeddings) == len(self.train_labels), f"Number of embeddings ({len(self.train_embeddings)}) must match number of labels ({len(self.train_labels)})"
        assert len(self.train_embeddings.shape) == 2, f"Embeddings must be 2D, got shape {self.train_embeddings.shape}"
        assert len(self.train_labels.shape) == 1, f"Labels must be 1D, got shape {self.train_labels.shape}"

        if self.centering:
            self.center = self.train_embeddings.mean(dim=0, keepdim=True)
            self.train_embeddings = self.train_embeddings - self.center

        if self.similarity == 'cosine' or self.centering:
            self.train_embeddings = F.normalize(self.train_embeddings.float(), dim=1)

    @torch.no_grad()
    def retrieve(self, queries, max_k):
        """
        Test the Retriever with test data (queries).

        Args:
            queries (np.ndarray): Query embeddings (shape: n x d).
            max_k (int): Maximum number of keys to retrieve for each query.

        Returns:
            topk_preds (torch.Tensor): Labels of top max_k retrieved keys for each query.
        """
        if self.centering:
            queries = queries - self.center

        if self.similarity == 'cosine' or self.centering:
            queries = F.normalize(queries.float(), dim=1)

        if self.similarity == 'cosine':
            sim_scores = torch.matmul(queries, self.train_embeddings.T)
        elif self.similarity == 'l2':
            sim_scores = -torch.cdist(queries, self.train_embeddings, p=2)  # Negative for similarity

        _, topk_ids = torch.topk(sim_scores, max_k, dim=1) # Shape: n x max(ks)

        topk_preds = self.train_labels[topk_ids] # Shape: n x max(ks)
        return topk_preds