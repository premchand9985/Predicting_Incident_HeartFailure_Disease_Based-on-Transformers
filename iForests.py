# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 19:24:59 2023

@author: premchand
"""

from collections import defaultdict
import io
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import PIL
import sklearn.datasets
import sklearn.preprocessing

import graphviz
n_points = 22
n_inliers = 10
n_outliers = 1
centers = [[2, 2], [-2, -2]]
cluster_std = [1, 1.5]

plots_dir = "./plots/"
def mkdirs(path):
    import os, errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
            
class iTree_Node():
    def __init__(self, parent, XX, features, indices, random_state):
        super().__init__()
        vmin = XX.min(axis=0)
        vmax = XX.max(axis=0)
        diff = vmax - vmin
        test = diff > 0
        if not np.any(test):
            split_index = None
            split_value = None
        else:
            split_index = random_state.choice(indices[test])
            split_value = random_state.uniform(
                low=vmin[split_index],
                high=vmax[split_index]
            )
        self.parent = parent
        self.features = features
        self.vmin = vmin
        self.vmax = vmax
        self.split_index = split_index
        self.split_value = split_value
        self.count = len(XX)
        if split_index is None:
            self.left = None
            self.right = None
        else:
            self.left = iTree_Node(
                self,
                XX=XX[XX[:,split_index] <= split_value],
                features=features,
                indices=indices,
                random_state=random_state
            )
            self.right = iTree_Node(
                self,
                XX=XX[XX[:,split_index] > split_value],
                features=features,
                indices=indices,
                random_state=random_state
            )
    
    def __str__(self):
        if self.parent is None:
            return f"Root ({self.count})"
        else:
            compare = "<=" if self.parent.left == self else ">"
            feature = self.features[self.parent.split_index]
            threshold = self.parent.split_value
            count = self.count
            return f"{feature} {compare} {threshold} ({count})"
    
    def __hash__(self):
        return id(self)
    
    
class iTree():
    def __init__(self, XX, features=None, random_state=None):
        super().__init__()
        XX = np.asarray(XX)
        feature_indices = np.arange(XX.shape[1])
        if features is None:
            features = feature_indices
        features = np.asarray(features)
        assert(XX.shape[1] == len(features))
        self.features = features
        self.root = iTree_Node(None, XX, features, feature_indices, random_state)
    
    def _traverse(self, node, tree):
        if node.split_index is not None:
            tree.setdefault(node, []).append(node.left)
            self._traverse(node.left, tree)
            
            tree.setdefault(node, []).append(node.right)
            self._traverse(node.right, tree)
            
    def _print_tree(self, output, parent, grandparent, tree, prefix, indent_width=2):
        output.write(str(parent) + "\n")
        if parent in tree:
            for ii,child in enumerate(tree[parent],1):
                if ii != len(tree[parent]):
                    s1, s2 = u"\u2560", u"\u2551"
                else:
                    s1, s2 = u"\u255A", " " 
                s3 = u"\u2550" if tree.get(child) == None else u"\u2566"
                output.write(u"{}{}{}{}".format(prefix, s1, u"\u2550" * indent_width, s3))
                new_prefix = u"{}{}{}".format(prefix, s2, " " * indent_width)
                self._print_tree(output, child, parent, tree, new_prefix, indent_width)
    
    def __str__(self):
        tree = {}
        self._traverse(i_tree.root, tree)
        output = io.StringIO()
        self._print_tree(
            output=output,
            parent=i_tree.root,
            grandparent=None,
            tree=tree,
            prefix="",
        )
        return output.getvalue()
    
    
def iTree2Digraph(i_tree, hidden=set(), node_attrs=dict()):
    def _traverse(node):
        if node.split_index is not None:
            parent = hex(id(node))
            attrs = dict(
                label=f"{node.features[node.split_index]}",
                color="white" if parent in hidden else "black",
                fontcolor="white" if parent in hidden else "black"
            )
            attrs.update(node_attrs.get(parent, dict()))
            graph.node(parent, **attrs)
            
            left = hex(id(node.left))
            attrs = dict(
                label=f"<= {node.split_value:0.2f}",
                color="white" if parent in hidden or left in hidden else "black",
                fontcolor="white" if parent in hidden or left in hidden else "black"
            )
            graph.edge(parent, left, **attrs)
            _traverse(node.left)
            
            right = hex(id(node.right))
            attrs = dict(
                label=f"> {node.split_value:0.2f}",
                color="white" if parent in hidden or right in hidden else "black",
                fontcolor="white" if parent in hidden or right in hidden else "black",
            )
            graph.edge(parent, right, **attrs)
            _traverse(node.right)
        else:
            parent = hex(id(node))
            attrs = dict(
                label=f"{node.count}",
                color="white" if parent in hidden else "black",
                fontcolor="white" if parent in hidden else "black"
            )
            attrs.update(node_attrs.get(parent, dict()))
            graph.node(parent, **attrs)
        
    graph = graphviz.Digraph()
    _traverse(i_tree.root)
    return graph


random = np.random.RandomState(0)

X_inliers, y_inliers = sklearn.datasets.make_blobs(
    n_samples=n_inliers,
    n_features=2,
    centers=centers,
    cluster_std=cluster_std,
    random_state=random
)

X_outliers = random.uniform(size=(n_outliers,2))

X_extra, y_extra = sklearn.datasets.make_blobs(
    n_samples=n_points,
    n_features=2,
    centers=centers,
    cluster_std=cluster_std,
    random_state=42
)

X_full = np.vstack([
    sklearn.preprocessing.MinMaxScaler((0.25,0.75)).fit_transform(X_extra),
    sklearn.preprocessing.MinMaxScaler((0.25,0.75)).fit_transform(X_inliers),
    sklearn.preprocessing.MinMaxScaler((0.1,0.9)).fit_transform(X_outliers),
])

X_sample = X_full[n_points:]
print(X_full.shape)
print(X_sample.shape)

fig,ax = plt.subplots(figsize=(16,10))
ax.scatter(*X_full.T, marker=".")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel("$x_{1}$")
ax.set_ylabel("$x_{2}$")
ax.set_aspect("equal")
plt.show()

fig,ax = plt.subplots(figsize=(16,10))
ax.scatter(*X_sample.T, marker=".")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel("$x_{1}$")
ax.set_ylabel("$x_{2}$")
ax.set_aspect("equal")
plt.show()