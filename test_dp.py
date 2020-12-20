import unittest


class MyTestCase(unittest.TestCase):

    def test_choice(self):
        from numpy.random import choice
        x = []
        for i in range(1000):
            x.append(choice([55,98], p=[0.9,0.1]))
        print(x)

    def test_poisson(self):
        from scipy.stats import poisson
        document_length = []
        for i in range(1000):
            document_length.append(poisson(5).rvs())
        print(document_length)

    def test_lda(self):
        vocabulary = ['see', 'spot', 'run']
        num_terms = len(vocabulary)
        num_topics = 2  # K
        num_documents = 5  # M
        mean_document_length = 5  # xi
        term_dirichlet_parameter = 1  # beta
        topic_dirichlet_parameter = 1  # alpha

        from scipy.stats import dirichlet, poisson
        from numpy import round
        from collections import defaultdict
        from random import choice as stl_choice
        term_dirichlet_vector = num_terms * [term_dirichlet_parameter]
        term_distributions = dirichlet(term_dirichlet_vector, 2).rvs(size=2)
        print(term_distributions)

        base_distribution = lambda: stl_choice(term_distributions)
        # A sample from base_distribution is a distribution over terms
        # Each of our two topics has equal probability
        from collections import Counter
        base_tuple = [tuple(base_distribution()) for _ in range(10000)]
        counter_base_tuple = Counter(base_tuple)
        counter_most_common = counter_base_tuple.most_common()
        for topic, count in counter_most_common:
            print("count:", count, "topic:", [round(prob, 2) for prob in topic])

        from scipy.stats import beta
        from numpy.random import choice
        class DirichletProcessSample():
            def __init__(self, base_measure, alpha):
                self.base_measure = base_measure
                self.alpha = alpha
                self.cache = []
                self.weights = []
                self.total_stick_used = 0.
            def __call__(self):
                remaining = 1.0 - self.total_stick_used
                i = DirichletProcessSample.roll_die(self.weights + [remaining])
                if i is not None and i < len(self.weights):
                    return self.cache[i]
                else:
                    stick_piece = beta(1, self.alpha).rvs() * remaining
                    self.total_stick_used += stick_piece
                    self.weights.append(stick_piece)
                    new_value = self.base_measure()
                    self.cache.append(new_value)
                    return new_value
            @staticmethod
            def roll_die(weights):
                if weights:
                    return choice(range(len(weights)), p=weights)
                else:
                    return None

        topic_distribution = DirichletProcessSample(base_measure=base_distribution,alpha=topic_dirichlet_parameter)
        for topic, count in Counter([tuple(topic_distribution()) for _ in range(10000)]).most_common():
            print("count:", count, "topic:", [round(prob, 2) for prob in topic])

        topic_index = defaultdict(list)
        documents = defaultdict(list)
        for doc in range(num_documents):
            topic_distribution_rvs = DirichletProcessSample(base_measure=base_distribution,alpha=topic_dirichlet_parameter)
            document_length = poisson(mean_document_length).rvs()
            for word in range(document_length):
                topic_distribution = topic_distribution_rvs()
                topic_index[doc].append(tuple(topic_distribution))
                documents[doc].append(choice(vocabulary, p=topic_distribution))


    def test_counter_most_common(self):
        words = [
            'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
            'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
            'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
            'my', 'eyes', "you're", 'under'
        ]
        from collections import Counter
        word_counts = Counter(words)
        top_three = word_counts.most_common(3)
        print(top_three)

    def test_hierarchy_dp(self):
        from numpy.random import choice
        from scipy.stats import beta
        from scipy.stats import norm
        import matplotlib.pyplot as plt
        from pandas import Series
        base_measure = lambda: norm().rvs()
        class DirichletProcessSample():
            def __init__(self, base_measure, alpha):
                self.base_measure = base_measure
                self.alpha = alpha
                self.cache = []
                self.weights = []
                self.total_stick_used = 0.
            def __call__(self):
                remaining = 1.0 - self.total_stick_used
                i = DirichletProcessSample.roll_die(self.weights + [remaining])
                if i is not None and i < len(self.weights):
                    return self.cache[i]
                else:
                    stick_piece = beta(1, self.alpha).rvs() * remaining
                    self.total_stick_used += stick_piece
                    self.weights.append(stick_piece)
                    new_value = self.base_measure()
                    self.cache.append(new_value)
                    return new_value
            @staticmethod
            def roll_die(weights):
                if weights:
                    return choice(range(len(weights)), p=weights)
                else:
                    return None

        class HierarchicalDirichletProcessSample(DirichletProcessSample):
            def __init__(self, base_measure, alpha1, alpha2):
                first_level_dp = DirichletProcessSample(base_measure, alpha1)
                self.second_level_dp = DirichletProcessSample(first_level_dp, alpha2)

            def __call__(self):
                return self.second_level_dp()

        norm_hdp = HierarchicalDirichletProcessSample(base_measure, alpha1=10, alpha2=20)
        for i in range(5):#the H is different every time we execute it
            norm_hdp = HierarchicalDirichletProcessSample(base_measure, alpha1=10, alpha2=10)
            _ = Series(norm_hdp() for _ in range(100)).hist()
            _ = plt.title("Histogram of samples from distribution drawn from Hierarchical DP")
            _ = plt.figure()
            plt.show()
        print('end')


    ############################################################

    def test_dp_class(self):
        import matplotlib.pyplot as plt
        from scipy.stats import beta, norm
        from numpy.random import choice
        import pandas as pd
        class DirichletProcessSample():
            def __init__(self, base_measure, alpha):
                self.base_measure = base_measure
                self.alpha = alpha
                self.cache = []
                self.weights = []
                self.total_stick_used = 0.
            def __call__(self):
                remaining = 1.0 - self.total_stick_used
                i = DirichletProcessSample.roll_die(self.weights + [remaining])
                if i is not None and i < len(self.weights):
                    return self.cache[i]
                else:
                    stick_piece = beta(1, self.alpha).rvs() * remaining
                    self.total_stick_used += stick_piece
                    self.weights.append(stick_piece)
                    new_value = self.base_measure()
                    self.cache.append(new_value)
                    return new_value
            @staticmethod
            def roll_die(weights):
                if weights:
                    return choice(range(len(weights)), p=weights)
                else:
                    return None

        base_measure = lambda: norm().rvs()
        n_samples = 10000
        samples = {}
        for alpha in [1000]:
            dirichlet_norm = DirichletProcessSample(base_measure=base_measure, alpha=alpha)
            samples["Alpha: %s" % alpha] = [dirichlet_norm() for _ in range(n_samples)]
        _ = pd.DataFrame(samples).hist()
        plt.show()

    def test_vline(self):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.vlines([0,2], 11, 18)
        # X = np.linspace(-4, 4, 100)
        # plt.plot(X, norm.pdf(X))
        plt.show()

    def test_beta(self):
        from scipy.stats import beta
        [a, b] = [0.6,0.6 ]
        bt = beta(1, 0.1)
        vs = []
        for i in range(1000):
            vs.append(bt.rvs())
        print('end')

    def test_dp(self):
        import matplotlib.pyplot as plt
        from scipy.stats import beta, norm
        import numpy as np
        def dirichlet_sample_approximation(base_measure, alpha, tol=0.01):
            betas = []
            pis = []
            betas.append(beta(1, alpha).rvs())
            pis.append(betas[0])
            while sum(pis) < (1. - tol):
                s = np.sum([np.log(1 - b) for b in betas])
                new_beta = beta(1, alpha).rvs()
                betas.append(new_beta)
                pis.append(new_beta * np.exp(s))
            pis = np.array(pis)
            thetas = np.array([base_measure() for _ in pis])#execute norm().rvs() n times:n is the size of pis
            return pis, thetas

        def plot_normal_dp_approximation(alpha):
            plt.figure()
            plt.title("Dirichlet Process Sample with N(0,1) Base Measure")
            plt.suptitle("alpha: %s" % alpha)
            pis, thetas = dirichlet_sample_approximation(lambda: norm().rvs(), alpha)
            pis = pis * (norm.pdf(0) / pis.max())#normalize pis
            plt.vlines(thetas, 0, pis, )
            X = np.linspace(-4, 4, 100)#[-4,4],with 100 steps
            plt.plot(X, norm.pdf(X))
            plt.show()
            print('end')

        plot_normal_dp_approximation(1000)
        print('end')
        # plot_normal_dp_approximation(1)
        # plot_normal_dp_approximation(10)
        # plot_normal_dp_approximation(1000)

    def test_dd_single(self):
        import numpy as np
        from scipy.stats import dirichlet
        np.set_printoptions(precision=2)
        samples = dirichlet(alpha=1000 * np.array([0.1,0.3,0.6])).rvs(10000)
        print("element-wise mean:", samples.mean(axis=0))
        print("element-wise standard deviation:", samples.std(axis=0))
        print('end')

    def test_dd(self):
        import numpy as np
        from scipy.stats import dirichlet
        np.set_printoptions(precision=2)
        def stats(scale_factor, G0=[0.2, 0.2, 0.6], N=10000):
            samples = dirichlet(alpha=scale_factor * np.array(G0)).rvs(N)
            print("alpha:", scale_factor)
            print("element-wise mean:", samples.mean(axis=0))
            print("element-wise standard deviation:", samples.std(axis=0))
            print()
        for scale in [0.1, 1, 10, 100, 1000]:
            stats(scale)


if __name__ == '__main__':
    unittest.main()
