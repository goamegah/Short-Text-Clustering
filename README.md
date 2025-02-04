<h1 align="center">⚡️ short-text-clustering </h1>

<h4 align="center">
    <p>
        <a href="#Tandem-approach-for-short-text-clustering">Tandem approach for short text clustering</a> •
        <a href="#Installation">Installation</a> •
        <a href="#Experiments-and-results">Experiments and results</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://www.iledefrance.fr/"><img style="float: middle; padding: 10px 10px 10px 10px;" width="400" height="260" src="assets/clust.png" /></a>
</h3>


# Tandem approach for short text clustering

The tandem approach in clustering combines dimensionality reduction, which reduces the complexity of
data while retaining relevant information, with clustering, which groups similar data point

# Installation

- Create virtual environment 

We are considering **venv** but feel free to other tools available.

**Note**: based on what is publicly available, you might need (for python users), package **spherecluster**. The package is currently being updated. Nevertheless, you can follow the instructions to see the program running.

```
$ python -m venv tdmenv
$ source tdmenv/bin/activate
$ pip install -r requirements.txt
```

# Experiments and results

Let's check results in table below, biomedical, stackoverflow and searchsnippets (top to bottom resp.). We apply on  HuggingFace embedding () UMAP dimensionality reduction followed by clustering algorithm like skmeans++ or sphérical-kmeans++. Important results are highlighted.


## Spherical kmeans++

![spherical kmeans result](assets/SPHERICAL.png)

## kmeans++

![kmeans++ results](assets/KMEANS_PP.png)

## Visualization

![kmeans++ results](assets/tandem_viz.png)