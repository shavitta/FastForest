import math
import random
from logging import warn

import numpy as np
from joblib import Parallel, delayed
from pandas.core.dtypes.common import is_string_dtype

from scipy.sparse import issparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble._forest import MAX_INT, _parallel_build_trees, _get_n_samples_bootstrap
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._tree import DOUBLE, issparse, DTYPE
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight

VERBOSE = False


# A class to represent a node in the classifying tree
class Node:
    def __init__(self, left_child=None, right_child=None, feature=-1, threshold=-1, impurity=-1, n_node_samples=0,
                 probs=None, is_leaf=False):
        self.left_child = left_child  # Pointer to the left sub tree
        self.right_child = right_child  # Pointer to the right sub tree
        self.feature = feature  # The feature according to which the split is made
        self.threshold = threshold  # The values according to which the split is made
        self.impurity = impurity  # The impurity index of this node
        self.n_node_samples = n_node_samples  # The number of samples in the node
        self.probs = probs  # The prob to belong to each class in this node. relevant only for leaves
        self.is_leaf = is_leaf  # Indication whether this node is a leaf


class MyDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, *,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 ccp_alpha=0.0):
        super().__init__(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            ccp_alpha=ccp_alpha)

        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        if VERBOSE:
            print("In MyDecisionTreeClassifier, max_features={}".format(self.max_features))

    def get_params(self, deep=True):
        return {"criterion": self.criterion,
                "splitter": self.splitter,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
                "max_features": self.max_features,
                "class_weight": self.class_weight,
                "random_state": self.random_state,
                "min_impurity_decrease": self.min_impurity_decrease,
                "min_impurity_split": self.min_impurity_split,
                "ccp_alpha": self.ccp_alpha}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted="deprecated"):
        if VERBOSE:
            print("in MyDecisionTreeClassifier fit function :::: X size: {}, y size: {}".format(len(X), len(y)))
        self.depth = 0
        self.n_classes_ = int(np.max(y)) + 1
        self.n_samples_, self.n_features_ = X.shape

        self.root = self.build_tree(X, y)
        self.print_tree(self.root)
        return self

    def is_node_leaf(self, y):
        if VERBOSE:
            print(
                "In is_node_leaf ::: depth, max_depth, min_samples_leaf= {}, {}, {}".format(self.depth, self.max_depth,
                                                                                            self.min_samples_leaf))
            print("np.unique(y): {}, len(y): {}".format(np.unique(y), len(y)))

        # check if there is only 1 class in the results --> got a classification
        if len(np.unique(y)) == 1:
            return True
        # check if the number of samples in the node is at least twice min_samples_leaf
        if len(y) < 2 * self.min_samples_leaf:
            return True
        # check the tree depth is still less than the max_depth
        if self.depth >= self.max_depth:
            return True
        return False

    def get_classification_proba(self, y):
        unique_classes, classes_count = np.unique(y, return_counts=True)
        probs = np.zeros(self.n_classes_, dtype=np.float)

        for i in range(len(classes_count)):
            c = int(unique_classes[i])
            try:
                probs[c] = classes_count[i] / y.shape[0]
            except IndexError:
                print(unique_classes, classes_count, y)
        if VERBOSE:
            print("In get_classification_proba ::: self.n_classes_={}".format(self.n_classes_))
            print("unique_classes {}, classes_count {}, probs {}".format(unique_classes, classes_count, probs))
        return probs

    def get_potential_values_for_split(self, unique_vals, is_nominal, n_samples_in_node):
        unique_vals = np.sort(unique_vals)
        if VERBOSE:
            print("In get_potential_values_for_split ::: unique_vals {}, is_nominal {}, n_samples_in_node {}".format(
                unique_vals, is_nominal, n_samples_in_node))
        if len(unique_vals) <= 1:
            return []

        # If nominal feature then need to return all values
        if is_nominal:
            if VERBOSE:
                print("Nominal case - returning all unique_vals {}".format(unique_vals))
            return unique_vals

        if self.max_features == 'fast':
            # FastForest case: if the feature if numeric then need to check in hops of log2(n_samples in the node) only
            hop_start = unique_vals[0]
            value_range = unique_vals[-1] - hop_start
            num_hops = int(np.round(np.log2(n_samples_in_node)) + 1)
            hop_step = value_range / num_hops
            vals = [hop_start + i * hop_step for i in range(num_hops)]
            if VERBOSE:
                print("Fast case - returning {}".format(vals))
            return vals

        # "Regular" case - get all values but the first one
        if VERBOSE:
            print("Regular case - returning {}".format(unique_vals[1:len(unique_vals)]))
        return unique_vals[1:len(unique_vals)]

    def split(self, X, y, c_i, t, is_nominal):
        if VERBOSE:
            print("In split ::: c_i, t={}, {}".format(c_i, t))
        right_indices = []
        left_indices = []
        if is_nominal:
            left_indices = [X[:, c_i] == t]
            right_indices = [X[:, c_i] != t]
        else:
            left_indices = [X[:, c_i] <= t]
            right_indices = [X[:, c_i] > t]
        return X[right_indices], y[right_indices], X[left_indices], y[left_indices]

    def calc_entropy(self, y):
        unique_classes, classes_count = np.unique(y, return_counts=True)
        probs = classes_count / classes_count.sum()
        entropy = sum(probs * -np.log2(probs))
        if VERBOSE:
            print("entropy {}".format(entropy))
        return entropy

    def calculate_overall_entropy(self, y_right, y_left):
        n = len(y_right) + len(y_left)
        p_data_right = len(y_right) / n
        p_data_left = len(y_left) / n
        overall_entropy = (p_data_right * self.calc_entropy(y_right)
                           + p_data_left * self.calc_entropy(y_left))
        if VERBOSE:
            print("p_data_right {}, p_data_left {}, overall_entropy {}".format(p_data_right, p_data_left,
                                                                               overall_entropy))
        return overall_entropy

    def find_best_split(self, X, y, features_indices):
        if VERBOSE:
            print("In find_best_split ::: X.shape {}, features_indices {}".format(X.shape, features_indices))
        # go over all columns
        best_entropy = 100000000
        best_split_column = None
        best_split_value = None
        for i in range(len(features_indices)):
            is_nominal = is_string_dtype(X[:, features_indices[i]])
            values = self.get_potential_values_for_split(np.unique(X[:, features_indices[i]]), is_nominal, X.shape[0])
            if VERBOSE:
                print("Working on feature {}, is_nominal {}, values: {}, unique values: {}".format(features_indices[i],
                                                                                                   is_nominal,
                                                                                                   values,
                                                                                                   np.unique(X[:,
                                                                                                             features_indices[
                                                                                                                 i]])))
            # go over all possible values
            for value in values:
                X_right, y_right, X_left, y_left = self.split(X, y, features_indices[i], value, is_nominal)
                current_overall_entropy = self.calculate_overall_entropy(y_right, y_left)
                if VERBOSE:
                    print("Splitting by feature {}, value {}".format(features_indices[i], value))
                    print("X_right={}".format(X_right))
                    print("X_left={}".format(X_left))

                if current_overall_entropy <= best_entropy:
                    best_entropy = current_overall_entropy
                    best_split_column = features_indices[i]
                    best_split_value = value
                    if VERBOSE:
                        print("Found potential feature: {} {}".format(best_split_column, best_split_value))
        if VERBOSE:
            print("Returning best feature: {} {}".format(best_split_column, best_split_value))
        return best_split_column, best_split_value, best_entropy

    def build_tree(self, X, y):
        if self.is_node_leaf(y):
            probs = self.get_classification_proba(y)
            n = Node(is_leaf=True, probs=probs)
            if VERBOSE:
                print("In build_tree ::: node is leaf. its data: {}".format(n))
            return n

        self.depth += 1
        features_indices = self.get_subset_features_indices(X)

        best_split_column, best_split_value, best_entropy = self.find_best_split(X, y, features_indices)

        # if we didn't find any split then create a leaf
        if best_split_column is None:
            probs = self.get_classification_proba(y)
            n = Node(is_leaf=True, probs=probs)
            if VERBOSE:
                print("In build_tree ::: node is leaf. its data: {}".format(n))
            return n

        X_right, y_right, X_left, y_left = self.split(X, y, best_split_column, best_split_value,
                                                      is_string_dtype(X[:, best_split_column]))

        # If there is no data left for the split then create a leaf
        if (len(X_right) == 0) or (len(X_left) == 0):
            probs = self.get_classification_proba(y)
            n = Node(is_leaf=True, probs=probs)
            if VERBOSE:
                print("In build_tree ::: node is leaf. its data: {}".format(n))
            return n

        node = Node(feature=best_split_column, threshold=best_split_value, n_node_samples=X.shape[0],
                    impurity=best_entropy)
        if VERBOSE:
            print("Creating node {}".format(node))
        node.left_child = self.build_tree(X_left, y_left)
        node.right_child = self.build_tree(X_right, y_right)
        return node

    def get_subset_features_indices(self, X):
        if VERBOSE:
            print("In get_subset_features_indices ::: X.shape: {}, type: {}".format(X.shape, self.max_features))
        n_features_to_select = 0
        # First 2 cases are for the "regular" decision tree, the last for the FastForest algorithm
        if self.max_features == 'auto':
            # Need to randomly select sqrt(n_features) for X and return the new X
            n_features_to_select = np.round(np.sqrt(self.n_features_))
        elif self.max_features == 'log2':
            # Need to randomly select log2(n_features) for X and return the new X
            n_features_to_select = np.round(np.log2(self.n_features_))
        else:
            # FastForest case: need to select log2(n_features)+1 if number of samples in the node is bigger than
            # 0.125 of the original number of samples in the dataset. Otherwise, select log2(original n_features *
            # original n_samples / num samples in the node) + 1 and return the new X
            if VERBOSE:
                print("{} > 0.125 * {}".format(X.shape[0], self.n_samples_))
            if X.shape[0] > 0.125 * self.n_samples_:
                n_features_to_select = np.round(np.log2(self.n_features_)) + 1
            else:
                n_features_to_select = int(np.round(np.log2(self.n_features_ * self.n_samples_ / X.shape[0])))
                n_features_to_select = min(self.n_features_, n_features_to_select)

        # Select n_features_to_select indices randomly
        if VERBOSE:
            print("X.shape[1] {}, int(n_features_to_select {}".format(X.shape[1], int(n_features_to_select)))
        feature_indices = random.sample(range(X.shape[1]), int(n_features_to_select))
        if VERBOSE:
            print("feature_indices:{}".format(feature_indices))
        return feature_indices

    def print_tree(self, node, i=0):
        if VERBOSE:
            if node.is_leaf:
                print("\t" * i + "LEAF, probs: {}".format(node.probs))
            else:
                print("\t" * i + "X[{}] < {}".format(node.feature, node.threshold))
                i += 1
                if node.left_child != None:
                    self.print_tree(node.left_child, i)
                if node.right_child != None:
                    self.print_tree(node.right_child, i)

    def predict_proba(self, X, check_input=True):
        probs = []
        for sample in X:
            res = self.predict_single(sample, self.root)
            if VERBOSE:
                print("sample: {}, res: {}".format(sample, res))
            probs.append(res)
        return probs

    def predict_single(self, sample, node):
        if VERBOSE:
            print("In predict_single ::: sample={}, node={}".format(sample, node))
        if node.is_leaf:
            if VERBOSE:
                print("Returning: {}".format(node.probs))
            return node.probs

        feature_index = node.feature
        threshold = node.threshold
        is_nominal = is_string_dtype(sample[feature_index])

        if is_nominal:
            if sample[feature_index] == threshold:
                return self.predict_single(sample, node.left_child)
            else:
                return self.predict_single(sample, node.right_child)
        else:
            if sample[feature_index] < threshold:
                return self.predict_single(sample, node.left_child)
            else:
                return self.predict_single(sample, node.right_child)


def _parallel_build_trees_with_half_sub_bagging(tree, forest, X, y, sample_weight, tree_idx, n_trees,
                                                verbose=0, class_weight=None,
                                                n_samples_bootstrap=None, random_state=33):
    if VERBOSE:
        print("in _parallel_build_trees_with_half_sub_bagging :::: X size: {}, y size: {}".format(len(X), len(y)))
        print("building tree %d of %d" % (tree_idx + 1, n_trees))

    # select half the sample randomly
    sample_indices = random.sample(range(X.shape[0]), math.floor(X.shape[0] / 2))
    new_X = X[sample_indices]
    new_y = y[sample_indices]

    if VERBOSE:
        print("in _parallel_build_trees_with_half_sub_bagging :::: new_X size: {}, new_y size: {}".format(len(new_X),
                                                                                                          len(new_y)))

    tree.fit(new_X, new_y, sample_weight=sample_weight, check_input=False)

    return tree


class FastForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        # X, y = self._validate_data(X, y, multi_output=True,
        #                           accept_sparse="csc", dtype=DTYPE)
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        # X = X.values
        # Check parameters
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            self.base_estimator_ = MyDecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                                            min_samples_split=self.min_samples_split,
                                                            min_samples_leaf=self.min_samples_split,
                                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                            max_features=self.max_features,
                                                            max_leaf_nodes=self.max_leaf_nodes,
                                                            min_impurity_decrease=self.min_impurity_decrease,
                                                            min_impurity_split=self.min_impurity_split,
                                                            random_state=self.random_state)

            if VERBOSE:
                print("n_more_estimators={}".format(n_more_estimators))
                print("self.base_estimator={}".format(self.base_estimator_))

            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees_with_half_sub_bagging)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight, n_samples_bootstrap=None)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self


class MyRandomForestClassifier(RandomForestClassifier):
    def __init__(self,
                 n_estimators=100, *,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, sample_weight=None):
        """
            Build a forest of trees from the training set (X, y).

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The training input samples. Internally, its dtype will be converted
                to ``dtype=np.float32``. If a sparse matrix is provided, it will be
                converted into a sparse ``csc_matrix``.

            y : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The target values (class labels in classification, real numbers in
                regression).

            sample_weight : array-like of shape (n_samples,), default=None
                Sample weights. If None, then samples are equally weighted. Splits
                that would create child nodes with net zero or negative weight are
                ignored while searching for a split in each node. In the case of
                classification, splits are also ignored if they would result in any
                single class carrying a negative weight in either child node.

            Returns
            -------
            self : object
            """
        # Validate or convert input data
        if issparse(y):
            raise ValueError(
                "sparse multilabel-indicator for y is not supported."
            )
        X, y = self._validate_data(X, y, multi_output=True,
                                   accept_sparse="csc", dtype=DTYPE)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)

        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        # Remap output
        self.n_features_ = X.shape[1]

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        y, expanded_class_weight = self._validate_y_class_weight(y)

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if expanded_class_weight is not None:
            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        # Get bootstrap sample size
        n_samples_bootstrap = _get_n_samples_bootstrap(
            n_samples=X.shape[0],
            max_samples=self.max_samples
        )

        # Check parameters
        self._validate_estimator()

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        random_state = check_random_state(self.random_state)

        if not self.warm_start or not hasattr(self, "estimators_"):
            # Free allocated memory, if any
            self.estimators_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
        else:
            self.base_estimator_ = MyDecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth,
                                                            min_samples_split=self.min_samples_split,
                                                            min_samples_leaf=self.min_samples_split,
                                                            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                                            max_features=self.max_features,
                                                            max_leaf_nodes=self.max_leaf_nodes,
                                                            min_impurity_decrease=self.min_impurity_decrease,
                                                            min_impurity_split=self.min_impurity_split,
                                                            random_state=self.random_state)
            if self.warm_start and len(self.estimators_) > 0:
                # We draw from the random state to get the random state we
                # would have got if we hadn't used a warm_start.
                random_state.randint(MAX_INT, size=len(self.estimators_))

            trees = [self._make_estimator(append=False,
                                          random_state=random_state)
                     for i in range(n_more_estimators)]

            # Parallel loop: we prefer the threading backend as the Cython code
            # for fitting the trees is internally releasing the Python GIL
            # making threading more efficient than multiprocessing in
            # that case. However, for joblib 0.12+ we respect any
            # parallel_backend contexts set at a higher level,
            # since correctness does not rely on using threads.
            trees = Parallel(n_jobs=self.n_jobs, verbose=self.verbose,
                             **_joblib_parallel_args(prefer='threads'))(
                delayed(_parallel_build_trees)(
                    t, self, X, y, sample_weight, i, len(trees),
                    verbose=self.verbose, class_weight=self.class_weight,
                    n_samples_bootstrap=n_samples_bootstrap)
                for i, t in enumerate(trees))

            # Collect newly grown trees
            self.estimators_.extend(trees)

        if self.oob_score:
            self._set_oob_score(X, y)

        # Decapsulate classes_ attributes
        if hasattr(self, "classes_") and self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self
