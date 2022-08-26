import numpy as np
import time as T

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.extmath import cartesian

import matplotlib.pyplot as plt


class MultivariateHawkes:
    def __init__(self, alpha=[[0.5]], mu=[0.1], omega=1.0):
        """params should be of form:
        alpha: numpy.array((u,u)), mu: numpy.array((,u)), omega: float"""

        self.data = []
        self.alpha, self.mu, self.omega = np.array(alpha), np.array(mu), omega
        self.dim = self.mu.shape[0]
        self.check_stability()

    def check_stability(self):
        """check stability of process (max alpha eigenvalue < 1)"""
        w, v = np.linalg.eig(self.alpha)
        me = np.amax(np.abs(w))
        print("Max eigenvalue: %1.5f" % me)
        if me >= 1.0:
            print("(WARNING) Unstable.")

    def generate_seq(self, horizon):
        """Generate a sequence based on mu, alpha, omega values.
        Uses Ogata's thinning method, with some speedups, noted below"""

        self.data = []  # clear history

        Istar = np.sum(self.mu)
        s = np.random.exponential(scale=1.0 / Istar)

        # attribute (weighted random sample, since sum(mu)==Istar)
        n0 = np.random.choice(np.arange(self.dim), 1, p=(self.mu / Istar))
        self.data.append([s, n0])

        # value of \lambda(t_k) where k is most recent event
        # starts with just the base rate
        lastrates = self.mu.copy()

        decIstar = False
        while True:
            tj, uj = self.data[-1][0], int(self.data[-1][1])

            if decIstar:
                # if last event was rejected, decrease Istar
                Istar = np.sum(rates)
                decIstar = False
            else:
                # otherwise, we just had an event, so recalc Istar (inclusive of last event)
                Istar = np.sum(lastrates) + self.omega * np.sum(self.alpha[:, uj])

            # generate new event
            s += np.random.exponential(scale=1.0 / Istar)

            # calc rates at time s (use trick to take advantage of rates at last event)
            rates = self.mu + np.exp(-self.omega * (s - tj)) * (
                self.alpha[:, uj].flatten() * self.omega + lastrates - self.mu
            )

            # attribution/rejection test
            # handle attribution and thinning in one step as weighted random sample
            diff = Istar - np.sum(rates)
            try:
                n0 = np.random.choice(np.arange(self.dim + 1), 1, p=(np.append(rates, diff) / Istar))
            except ValueError:
                # by construction this should not happen
                print("Probabilities do not sum to one.")
                self.data = np.array(self.data)
                return self.data

            if n0 < self.dim:
                self.data.append([s, n0])
                # update lastrates
                lastrates = rates.copy()
            else:
                decIstar = True

            # if past horizon, done
            if s >= horizon:
                self.data = np.array(self.data, dtype=float)
                self.data = self.data[self.data[:, 0] < horizon]
                return self.data

    # -----------
    # EM LEARNING
    # -----------

    def EM(
        self,
        Ahat,
        mhat,
        omega,
        seq=[],
        smx=None,
        tmx=None,
        regularize=False,
        Tm=-1,
        maxiter=100,
        epsilon=0.01,
        verbose=True,
    ):
        """implements MAP EM. Optional to regularize with `smx` and `tmx` matrix (shape=(dim,dim)).
        In general, the `tmx` matrix is a pseudocount of parent events from column j,
        and the `smx` matrix is a pseudocount of child events from column j -> i,
        however, for more details/usage see https://stmorse.github.io/docs/orc-thesis.pdf"""

        # if no sequence passed, uses class instance data
        if len(seq) == 0:
            seq = self.data

        N = len(seq)
        dim = mhat.shape[0]
        Tm = float(seq[-1, 0]) if Tm < 0 else float(Tm)
        sequ = seq[:, 1].astype(int)

        p_ii = np.random.uniform(0.01, 0.99, size=N)
        p_ij = np.random.uniform(0.01, 0.99, size=(N, N))

        # PRECOMPUTATIONS

        # diffs[i,j] = t_i - t_j for j < i (o.w. zero)
        diffs = pairwise_distances(np.array([seq[:, 0]]).T, metric="euclidean")
        diffs[np.triu_indices(N)] = 0

        # kern[i,j] = omega*np.exp(-omega*diffs[i,j])
        kern = omega * np.exp(-omega * diffs)

        colidx = np.tile(sequ.reshape((1, N)), (N, 1))
        rowidx = np.tile(sequ.reshape((N, 1)), (1, N))

        # approx of Gt sum in a_{uu'} denom
        seqcnts = np.array([len(np.where(sequ == i)[0]) for i in range(dim)])
        seqcnts = np.tile(seqcnts, (dim, 1))

        # returns sum of all pmat vals where u_i=a, u_j=b
        # *IF* pmat upper tri set to zero, this is
        # \sum_{u_i=u}\sum_{u_j=u', j<i} p_{ij}
        def sum_pij(a, b):
            c = cartesian([np.where(seq[:, 1] == int(a))[0], np.where(seq[:, 1] == int(b))[0]])
            return np.sum(p_ij[c[:, 0], c[:, 1]])

        vp = np.vectorize(sum_pij)

        # \int_0^t g(t') dt' with g(t)=we^{-wt}
        # def G(t): return 1 - np.exp(-omega * t)
        #   vg = np.vectorize(G)
        # Gdenom = np.array([np.sum(vg(diffs[-1,np.where(seq[:,1]==i)])) for i in range(dim)])

        k = 0
        old_LL = -10000
        START = T.time()
        while k < maxiter:
            Auu = Ahat[rowidx, colidx]
            ag = np.multiply(Auu, kern)
            ag[np.triu_indices(N)] = 0

            # compute m_{u_i}
            mu = mhat[sequ]

            # compute total rates of u_i at time i
            rates = mu + np.sum(ag, axis=1)

            # compute matrix of p_ii and p_ij  (keep separate for later computations)
            p_ij = np.divide(ag, np.tile(np.array([rates]).T, (1, N)))
            p_ii = np.divide(mu, rates)

            # compute mhat:  mhat_u = (\sum_{u_i=u} p_ii) / T
            mhat = np.array([np.sum(p_ii[np.where(seq[:, 1] == i)]) for i in range(dim)]) / Tm

            # ahat_{u,u'} = (\sum_{u_i=u}\sum_{u_j=u', j<i} p_ij) / \sum_{u_j=u'} G(T-t_j)
            # approximate with G(T-T_j) = 1
            if regularize:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (dim, dim)) + (smx - 1), seqcnts + tmx)
            else:
                Ahat = np.divide(np.fromfunction(lambda i, j: vp(i, j), (dim, dim)), seqcnts)

            if k % 10 == 0:
                try:
                    term1 = np.sum(np.log(rates))
                except:
                    print("Log error!")
                term2 = Tm * np.sum(mhat)
                term3 = np.sum(np.sum(Ahat[u, int(seq[j, 1])] for j in range(N)) for u in range(dim))
                # new_LL = (1./N) * (term1 - term2 - term3)
                new_LL = (1.0 / N) * (term1 - term3)
                if abs(new_LL - old_LL) <= epsilon:
                    if verbose:
                        print("Reached stopping criterion. (Old: %1.3f New: %1.3f)" % (old_LL, new_LL))
                    return Ahat, mhat
                if verbose:
                    print("After ITER %d (old: %1.3f new: %1.3f)" % (k, old_LL, new_LL))
                    print(" terms %1.4f, %1.4f, %1.4f" % (term1, term2, term3))

                old_LL = new_LL

            k += 1

        if verbose:
            print("Reached max iter (%d)." % maxiter)

        self.Ahat = Ahat
        self.mhat = mhat
        return Ahat, mhat

    # -----------
    # VISUALIZATION METHODS
    # -----------

    def get_rate(self, ct, d):
        # return rate at time ct in dimension d
        seq = np.array(self.data)
        if not np.all(ct > seq[:, 0]):
            seq = seq[seq[:, 0] < ct]
        return self.mu[d] + np.sum(
            [self.alpha[d, int(j)] * self.omega * np.exp(-self.omega * (ct - t)) for t, j in seq]
        )

    def plot_rates(self, horizon=-1):

        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        f, axarr = plt.subplots(
            self.dim * 2,
            1,
            sharex="col",
            gridspec_kw={"height_ratios": sum([[3, 1] for i in range(self.dim)], [])},
            figsize=(8, self.dim * 2),
        )
        xs = np.linspace(0, horizon, int((horizon / 100.0) * 1000))
        for i in range(self.dim):
            row = i * 2

            # plot rate
            r = [self.get_rate(ct, i) for ct in xs]
            axarr[row].plot(xs, r, "k-")
            axarr[row].set_ylim([-0.01, np.amax(r) + (np.amax(r) / 2.0)])
            axarr[row].set_ylabel("$\lambda(t)_{%d}$" % i, fontsize=14)
            r = []

            # plot events
            subseq = self.data[self.data[:, 1] == i][:, 0]
            axarr[row + 1].plot(subseq, np.zeros(len(subseq)) - 0.5, "bo", alpha=0.2)
            axarr[row + 1].yaxis.set_visible(False)

            axarr[row + 1].set_xlim([0, horizon])

        plt.tight_layout()

    def plot_events(self, horizon=-1, showDays=True, labeled=True):
        if horizon < 0:
            horizon = np.amax(self.data[:, 0])

        fig = plt.figure(figsize=(10, 2))
        ax = plt.gca()
        for i in range(self.dim):
            subseq = self.data[self.data[:, 1] == i][:, 0]
            plt.plot(subseq, np.zeros(len(subseq)) - i, "bo", alpha=0.2)

        if showDays:
            for j in range(1, int(horizon)):
                plt.plot([j, j], [-self.dim, 1], "k:", alpha=0.15)

        if labeled:
            ax.set_yticklabels("")
            ax.set_yticks(-np.arange(0, self.dim), minor=True)
            ax.set_yticklabels([r"$e_{%d}$" % i for i in range(self.dim)], minor=True)
        else:
            ax.yaxis.set_visible(False)

        ax.set_xlim([0, horizon])
        ax.set_ylim([-self.dim, 1])
        ax.set_xlabel("Days")
        plt.tight_layout()
