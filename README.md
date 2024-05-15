<h1 align="center">⚡️ short-text-clustering </h1>

<h4 align="center">
    <p>
        <a href="#Objectif">Objectif</a> •
        <a href="#Les-fonctionnalités">Les fonctionnalités</a> •
        <a href="#Lancement">Lancement</a> •
    <p>
</h4>

<h3 align="center">
    <a href="https://www.iledefrance.fr/"><img style="float: middle; padding: 10px 10px 10px 10px;" width="250" height="160" src="assets/clust.png" /></a>
</h3>


# Tandem approach for short text clustering

The tandem approach in clustering combines dimen-
sionality reduction, which reduces the complexity of
data while retaining relevant information, with clus-
tering, which groups similar data point

# Installation


# Results

Let's check results in table below, biomedical, stackoverflow and searchsnippets (top to bottom resp.). We apply on  HuggingFace embedding () UMAP dimensionality reduction followed by clustering algorithm like skmeans++ or sphérical-kmeans++. Important results are highlighted.


## Spherical kmeans++

![spherical kmeans result](assets/SPHERICAL.png)

## kmeans++

![kmeans++ results](assets/KMEANS_PP.png)
