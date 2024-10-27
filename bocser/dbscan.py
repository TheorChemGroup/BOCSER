import numpy as np

class DBSCAN:
    
    def __init__(
        self,
        eps : float = 0.5,
        min_pts : int = 3,
        period : float = 2*np.pi
    ):
        """
            Simple DBSCAN algo
        """
        
        self.eps = eps
        self.min_pts = min_pts
        self.num_of_clusters = 0
        self.period = period
        
    def euclidean_dist(
        self, 
        a : np.ndarray, 
        b : np.ndarray
    ) -> float:
        return np.sqrt((a - b).dot(a - b))

    def max_angle_diff_dist(
        self, 
        a : np.ndarray, 
        b : np.ndarray
    ) -> float:
        dists = np.abs(a - b) % self.period
        return np.max(
            np.min(
                [dists, self.period-dists],
                axis=0
            )
        )
        
    def fit_predict(
        self, 
        X : np.ndarray
    ) -> np.ndarray:
        """
            Performs DBSCAN algorithm, returns labels for each point from X
        """
        self.labels_ = -np.ones(X.shape[0], int)
        
        # Iterate through all points
        for i in range(X.shape[0]):
            if self.labels_[i] != -1:
                continue
            # Find neighbors in 'eps'     
            neighbors = []
            for j in range(X.shape[0]):
                if i == j:
                    continue
                if self.max_angle_diff_dist(X[i], X[j]) <= self.eps:
                    neighbors.append(j)
                    
            if (len(neighbors) + 1) < self.min_pts:
                continue
            self.labels_[i] = self.num_of_clusters
            for j in neighbors:
                if self.labels_[j] != -1:
                    continue
                self.labels_[j] = self.num_of_clusters
                new_neighbors = []
                for k in range(X.shape[0]):
                    if k == j:
                        continue
                    if self.max_angle_diff_dist(X[j], X[k]) <= self.eps:
                        new_neighbors.append(k)
                if (len(new_neighbors) + 1) >= self.min_pts:
                    neighbors.extend(new_neighbors)
            self.num_of_clusters += 1
            
        return self.labels_
