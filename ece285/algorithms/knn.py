"""
K Nearest Neighbours Model
"""
import numpy as np


class KNN(object):
    def __init__(self, num_class: int):
        self.num_class = num_class

    def train(self, x_train: np.ndarray, y_train: np.ndarray, k: int):
        """
        Train KNN Classifier

        KNN only need to remember training set during training

        Parameters:
            x_train: Training samples ; np.ndarray with shape (N, D)
            y_train: Training labels  ; snp.ndarray with shape (N,)
        """
        self._x_train = x_train
        self._y_train = y_train
        self.k = k

    def predict(self, x_test: np.ndarray, k: int = None, loop_count: int = 1):
        """
        Use the contained training set to predict labels for test samples

        Parameters:
            x_test    : Test samples                                     ; np.ndarray with shape (N, D)
            k         : k to overwrite the one specificed during training; int
            loop_count: parameter to choose different knn implementation ; int

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # Fill this function in
        k_test = k if k is not None else self.k
        N, D = x_test.shape

        if loop_count == 1:
            distance = self.calc_dis_one_loop(x_test)
        elif loop_count == 2:
            distance = self.calc_dis_two_loop(x_test)

        # TODO: implement me
        # pass
        out = np.zeros(N)
        M, _ = self._x_train.shape
        
        for i in range(N):
            dis_class_pair = []
            
            for j in range(M):
                dis_class_pair.append([distance[i][j], self._y_train[j]])
            
            dis_class_pair.sort()
            class_list = []
            
            for n in range(k_test):
                class_list.append(dis_class_pair[n][1])

            out[i] = max(class_list, key=class_list.count)
            
        return out
            

    def calc_dis_one_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could one for loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """

        # TODO: implement me
        # pass
        N, D = x_test.shape
        M, _ = self._x_train.shape

        distances = np.zeros((N, M)) # Initialize array to store distances
        distances_class_list = []
        
        # Loop over each sample and calculate distances
        for i in range(N):
            distances[i] = np.linalg.norm(self._x_train - x_test[i], axis=1) # Euclidean distance
        
        return distances

    def calc_dis_two_loop(self, x_test: np.ndarray):
        """
        Calculate distance between training samples and test samples

        This function could contain two loop

        Parameters:
            x_test: Test samples; np.ndarray with shape (N, D)
        """
        # TODO: implement me
        # pass
        N, D = x_test.shape
        M, _ = self._x_train.shape

        distances = np.zeros((N, M)) # Initialize array to store distances

        # Loop over each test sample and training sample to calculate distances
        for i in range(N):
            for j in range(M):
                distances[i, j] = np.linalg.norm(x_test[i] - self._x_train[j]) # Euclidean distance

        return distances