import numpy as np

class CustomHMM:
    """
    Implémentation d'un modèle de Chaîne de Markov Cachée (HMM) en Python.
    """

    def __init__(self, n_states, n_observations):
        """
        Initialise le modèle HMM.

        :param n_states: Nombre d'états cachés
        :param n_observations: Nombre de symboles observables
        """
        self.n_states = n_states
        self.n_observations = n_observations

        # Initialisation aléatoire des matrices A, B et π
        self.A = np.random.rand(n_states, n_states)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.B = np.random.rand(n_states, n_observations)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.random.rand(n_states)
        self.pi /= self.pi.sum()

    def forward(self, observations):
        """
        Implémente l'algorithme Forward.

        :param observations: Séquence d'observations (liste d'indices)
        :return: alpha (matrice des probabilités avant) et probabilité totale
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # Initialisation
        alpha[0] = self.pi * self.B[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self.B[j, observations[t]]

        # Probabilité totale de la séquence observée
        prob = np.sum(alpha[T-1])
        return alpha, prob

    def backward(self, observations):
        """
        Implémente l'algorithme Backward.

        :param observations: Séquence d'observations
        :return: beta (matrice des probabilités arrière) et probabilité totale
        """
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialisation
        beta[T-1] = 1

        # Induction
        for t in reversed(range(T-1)):
            for i in range(self.n_states):
                beta[t, i] = np.sum(self.A[i] * self.B[:, observations[t+1]] * beta[t+1])

        # Probabilité totale
        prob = np.sum(self.pi * self.B[:, observations[0]] * beta[0])
        return beta, prob

    def viterbi(self, observations):
        """
        Implémente l'algorithme de Viterbi pour trouver la séquence d'états la plus probable.

        :param observations: Séquence d'observations
        :return: chemin optimal et probabilité associée
        """
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # Initialisation
        delta[0] = self.pi * self.B[:, observations[0]]

        # Induction
        for t in range(1, T):
            for j in range(self.n_states):
                probas = delta[t-1] * self.A[:, j]
                psi[t, j] = np.argmax(probas)
                delta[t, j] = np.max(probas) * self.B[j, observations[t]]

        # Chemin optimal
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])

        for t in reversed(range(T-1)):
            states[t] = psi[t+1, states[t+1]]

        prob = np.max(delta[T-1])
        return states, prob

    def baum_welch(self, observations, n_iter=100):
        """
        Implémente l'algorithme de Baum-Welch pour entraîner le modèle.

        :param observations: Séquence d'observations
        :param n_iter: Nombre d'itérations
        """
        T = len(observations)

        for n in range(n_iter):
            alpha, prob_fwd = self.forward(observations)
            beta, _ = self.backward(observations)

            # Calcul de gamma et xi
            gamma = np.zeros((T, self.n_states))
            xi = np.zeros((T-1, self.n_states, self.n_states))

            for t in range(T):
                denom = np.sum(alpha[t] * beta[t])
                gamma[t] = (alpha[t] * beta[t]) / denom

            for t in range(T-1):
                denom = np.sum(alpha[t] * np.dot(self.A, self.B[:, observations[t+1]] * beta[t+1]))
                for i in range(self.n_states):
                    numer = alpha[t, i] * self.A[i] * self.B[:, observations[t+1]] * beta[t+1]
                    xi[t, i] = numer / denom

            # Mise à jour des paramètres
            self.pi = gamma[0]

            for i in range(self.n_states):
                for j in range(self.n_states):
                    numer = np.sum(xi[:, i, j])
                    denom = np.sum(gamma[:-1, i])
                    self.A[i, j] = numer / denom

            for i in range(self.n_states):
                for k in range(self.n_observations):
                    mask = (np.array(observations) == k)
                    numer = np.sum(gamma[mask, i])
                    denom = np.sum(gamma[:, i])
                    self.B[i, k] = numer / denom

            # Normalisation
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.B /= self.B.sum(axis=1, keepdims=True)

