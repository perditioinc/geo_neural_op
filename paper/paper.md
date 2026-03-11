---
title: 'GeoNeuralOp: Operator Learning for Geometric Tasks and Point-Clouds.'
tags:
  - Python
  - geometric deep learning
  - neural operators
  - meshless methods
authors:
  - name: Blaine Quackenbush
    orcid: 0009-0005-5414-9008
    affiliation: 1
  - name: Paul J. Atzberger
    orcid: 0000-0001-6806-8069
    affiliation: 1
affiliations:
 - name: University of California Santa Barbara
   index: 1
date: 26 Jan 2026
bibliography: paper.bib
---

# Summary

The package [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) provides
methods for working with Geometric Neural Operators (GNPs).  The approaches
allow users to process geometric data without the need for explicit mesh
generation and facilitates estimating high-order geometric quantities,
differential operators, and other features from point-cloud data.  The package
includes training methods and transferable pre-trained models for geometric
tasks on oriented point-cloud manifold representations.  Capabilities include
approaches for (i) learning robust estimators for local metrics and curvatures,
(ii) estimating geometric differential operators, (iii) learning solution maps
for geometric partial differential equations (PDEs), (iv) computing geometric
flows such as mean-curvature driven shape deformations, and (v) performing other geometric
tasks.  The package also includes pre-trained weights that can serve as
transferable models for use within other machine learning approaches and
numerical methods.

We discuss here primarily the computational methods and approaches
implemented in the software package [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op). 
For a more technical discussion of our operator learning methods and results, please see 
[@quackenbush_atzberger_gnp_transfer_2025],
[@quackenbush_atzberger_gnps_2024], and [@quackenbush_atzberger_extension_2026]. 
We also discuss for our package additional algorithmic and computational considerations 
for making the methods more efficient. This includes approaches for using a few different 
types of kernel integration approaches utilizing block factorizations or 
separable representations. These computational methods allow for additional performance 
and efficiency gains facilitating scalability of training, inference, and also when 
deploying transferable pre-trained GNP models.  The package provides methods that can be 
used for diverse types of geometric processing with capabilities 
to handle arbitrary shapes and topologies.


# Statement of need

Geometric tasks involving point-cloud representations arise in many
applications. This includes in computer vision, robotics, control, geology, and
scientific computation (cite). The pointwise sampled data and noise often poses
challenges for estimation of geometric quantities and more global properties
associated with shape and topology.  This includes reliably obtaining accurate
estimates of local surface normals, curvature tensors, differential operators,
and solutions of geometric partial differential equations.  Our recent work on
Geometric Neural Operators (GNPs) addresses challenges on these issues by
leveraging operator learning approaches that incorporate inductive biases for
representing and extracting geometric information
[@quackenbush_atzberger_gnps_2024], [@quackenbush_atzberger_gnp_transfer_2025],
and [@quackenbush_atzberger_extension_2026]. Our methods provide meshless
data-driven techniques for geometric estimation and learning operators on
manifolds have general shapes and topologies.  

The [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) package provides
methods for Geometric Neural Operators (GNPs) to train data-driven operators
for estimating geometric quantities, computing differential operators, 
learning solution maps for PDEs, and other tasks on point-cloud data.  The
package also includes pre-trained transferable basal foundation models 
that can be used to perform geometric tasks in other machine learning 
pipelines or numerical methods
[@quackenbush_atzberger_gnp_transfer_2025].  In contrast to hand-crafted
estimators, the approaches provide data-driven ways to achieve robustness to
noise and to obtain efficiencies for more scalable geometric methods. The
approaches have been demonstrated for tasks in geometric estimation, solving
geometric PDEs, and shape deformations by curvature-driven flows
[@quackenbush_atzberger_gnps_2024],[@quackenbush_atzberger_gnp_transfer_2025],
and [@quackenbush_atzberger_extension_2026]. 

# State of the field

There are many general packages for working with geometry, including
`PytorchGeometric` for geometric training of graph neural networks
[@PyTorch-Geometric], `Geomstats` for statistics on Riemannian manifolds
[@miolane2020geomstats], `PyRiemann` for learning on Riemannian manifolds
[@barachant2011multiclass] [@pyriemann], `coxeter` for generating and handling
different shapes and topology [@Ramasubramani2021],
'PDAL' for processing point clouds [@butler2021pdal], and 'MeshLib' for handling
meshed geometries [@meshlib].  In addition more specialized packages for
specific application domains have also been developed, including for vascular
geometries [@izzo2018vascular] [@kjeldsberg2019morphman],
[@kjeldsberg2023vampy], for geometric engineering design [@hajdik2023pygeo],
and other applications [@diggins2026pymetric] [@wang2023pypose]
[@dewez2016facets] [@Neuromancer2023] [@girardeau2016cloudcompare].  While
these packages can be used to support operations for learning on manifolds they
are primarily designed to be used once the geometry is known. 

Our work addresses a different set of problems concerned with the upstream
tasks of learning the manifold representations more directly from the data and
the downstream tasks of data-driven learning of operators on these manifolds.
Current popular packages for related processing of geometric data include
Open3D [@Zhou2018] and PCL [@Rusu_ICRA2011_PCL]. While these provide 
fundamental methods for point-cloud processing, they are primarily limited to 
first-order geometric approximations to quantities such as the surface normals.  
For higher-order
geometric estimations, techniques such as in the package Jet Fitting using CGAL
[@cgal] or the package for Generalized Moving Least Squares (GMLS) [@compadre]
are capable of accurate estimations but can be sensitive to noise and outliers. 

The current data-driven methods for geometric tasks also use general neural
network architectures containing minimal inductive biases designed to handle
geometry. As a consequence, the resulting models often have a significant
dependence on specific manifold shapes and training class. The methods also
further are trained targeting specific geometric quantities on a specific set
of manifold shapes instead of aiming to extract more generic local geometric
information. While this results in useful specialized tools for target tasks
this also limits the scope of tasks for which the methods can be transferred. 



## Geometric Neural Operators
\label{gnp}

Geometric Neural Operators (GNPs) [@quackenbush_atzberger_gnps_2024] 
learning non-linear mappings between infinite dimensional function 
spaces on manifolds. They further develop neural network architectures 
that are based on kernel operator layers introduced in [@Kovachki2023a].
Through design of these architectures they are able to learn 
mappings between numerical representations of functions 
that are less sensitive to the underlying discretization resolution 
in comparison to more traditional neural networks, 
such as multi-layer perceptrons (MLPs) or convolutional 
neural networks (CNNs). 
GNPs also are designed to provide ways to work with meshless 
unstructured data and for incorporating 
further information from the geometric contributions of the 
underlying domain of the functions. Together these provide robust ways to 
learn diverse operators mapping between functions on manifolds having 
general shapes and topologies.

![
**Geometric Neural Operators.** We show the neural network architecture used
for capturing geometric features and performing tasks. To achieve additional
computational efficiencies, we also  provide a few different integral operator 
  implementations including (i) full integrations, (ii) block-factorized, and 
(iii) separable kernels. Image courtesy of 
  [@quackenbush_atzberger_gnp_transfer_2025].
  ](fig/gnp.png){ width=100% }

The GNP architecture consists of the followings parts
(i) a lifting layer that maps features to a higher dimension latent space,
(ii) one or more compositions of a linear operator layer followed by a 
non-linear activation, and (iii) a projection layer that projects the latent 
features into the desired output space. The linear operator layers consist of 
a pointwise linear operator (matrix) $W$, an integral operator $K[\cdot]$, and a 
bias function $b(x)$. The integral operator takes the form of integration against a learned 
kernel $k=k_\theta(\cdot,\cdot)$, with 
$$K[v](x) = \int k(x, y) v(y) \ dy.$$
We use $k(x, y) \in \mathbb{R}^{d_o} \times \mathbb{R}^{d_i}$ a matrix-valued
mapping of the function values $v(y) \in \mathbb{R}^{d_i}$ 
from $\mathbb{R}^{d_i}$ to $\mathbb{R}^{d_o}$. By composing the $T$ operator 
layers, we obtain the architecture $$G[f] = Q \circ \sigma (W_T + K_T
+ b_T) \circ \dots \circ (W_1 + K_1 + b_1) \circ P [f].$$
The integral operator $K[\cdot]$ is computed on a restricted compact domain centered around 
the query point $x$, typically a ball of radius $r(x)$. This is implemented as edge-conditioned
graph convolution using PyTorch-Geometric [@PyTorch-Geometric]. We compute
for $K$ the integral operations at each points using the graph neighborhood of $x$. 

## Kernel integration: Computational Costs and Improving Efficiency
Computing a general kernel operator for $N$ points without any 
restrictions on $k(\cdot,\cdot)$ would involve approximating 
integrals that can have a computational cost of $O(N^2)$. In practice,
this can become prohibitively expensive both in GPU memory and compute 
time. The [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) package 
allows for more efficient alternatives by providing several variants of 
the GNP architectures that make different assumptions about the form
of the kernels.  For instance, the initial neural operator approach
used a kernel $k$ of the form of a 4-layer 
multilayer-perceptron (MLP) that outputs a dense matrix of shape $(d_o, d_i)$. 
In [@quackenbush_atzberger_gnps_2024] a more efficient alternative for 
GNP architectures was developed based on block-factorizations. It was shown
that this can significantly improve efficiency while retaining accuracy. 
We discuss below a few of these variants. For more details, see the 
[`GeoNeuralOp` package](https://github.com/atzberg/geo_neural_op).

### Block-factorized kernel integration
The block-factorized GNP variant assumes that the kernel takes a restricted form of
$k(x, y) = W_k \tilde k(x, y)$. It is further assumed that $\tilde k$ outputs a 
matrix that is block diagonal,
$$\tilde k(x, y) = 
\begin{bmatrix} 
B_1(x, y)   &  0  & 0 & 0 \\
0   & B_2(x, y) & 0 & 0\\
0   &  0  & \ddots & 0 \\
0   &  0  & 0 & B_c(x, y) 
\end{bmatrix}.$$
The block-factorized GNPs leverage depth to decrease memory usage and training 
time. They were found to be capable of maintaining a comparable level of 
accuracy as the unfactorized case, see [@quackenbush_atzberger_gnps_2024].

### Separable kernel integration
For additional efficiencies in evaluation and training, 
the [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) package.
also provides alternative implementations of GNPs that utilize a separable
kernel architecture of the form $k(x, y) = k_1(x) k_2(y)$. The separable kernel assumption 
offloads evaluation of the kernel networks to the nodes of the graph instead of the
edges. This provides further computational savings by allowing computations of the form
$$ K(v(x)) = k_1(x) \int k_2(y)v(y) \ dy. $$
Instead of evaluating the kernel 
network $k$ on all edges, the separable kernels need only be evaluated on 
the nodes, providing significant computational savings. This allows for 
computing the integral at each location $x$ more efficiently 
by using just gather-scatter operations from ``torch-scatter`` [@PyTorch-Geometric].
This also can be combined with other methods to achieve 
further efficiencies, such as taking the kernels $k_1, k_2$ 
either to be a standard or block-factorized kernel.

![
**Typical Data-Processing Pipeline.**  We show how Geometric Neural Operator (GNP) models can be
used to obtain robust geometric estimators using latent local patches. We also provide pre-trained GNPs 
and weights that can be used for geometric tasks within existing data-processing pipelines.
  Image courtesy of 
  [@quackenbush_atzberger_gnp_transfer_2025]
  ](fig/representation_learning.png){ width=100% }

# Software design

The [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) package was
developed in PyTorch [@PyTorch] and leverages methods from the
PyTorch-Geometric [@PyTorch-Geometric] ecosystems to integrate into existing
machine learning pipelines. The core models, ``GNP`` and ``PatchGNP``, and
their layers inherit from the PyTorch ``torch.nn.Module`` classes. This allows
users to easily incorporate geometric neural operators into larger neural
network architectures. This also allows for utilizing other standard PyTorch
training and optimization loops. The methods also can be used to perform
standalone inference on CPUs or GPUs.  

In our package, we provide several GNP architectures in the class
``gnp.models``.  This includes the base ``GNP`` architecture and the
``PatchGNP`` model for which we provide pre-trained weights.  The
``gnp.models.GNP`` can be configured with different graph convolution types as
mentioned in Section \ref{gnp}. These types include a full
``GraphConvolution``, reduced ``BlockFactorizedConvolution``, and reduced
``SeparableConvolution``. 

Our package also provides transferable pre-trained basal foundation models
[@quackenbush_atzberger_gnp_transfer_2025]. These pre-trained models are
included within the library and can be automatically loaded by users with a
high-level API in the class ``GeometryEstimator``.  This enables immediate
access to robust geometric estimation without the need for technical knowledge
or extensive retraining of the weights of the GNPs.  The
[`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) with notation v1.x.x
contains the original pre-trained models and implementations from
[@quackenbush_atzberger_gnp_transfer_2025]. We use notation v2.x.x to
distinguish our more recent work that uses the optimized data structure
`PatchData` and for the pre-trained models based on these more efficient
separable kernel architectures.  See the repository for the package for 
the latest versions and implementations.  

**Geometric Estimators:** Our pre-trained GNP models and training methods for
the geometric estimators also allow for coping with noise and other 
artifacts that arise when processing point-clouds in practice.  This 
allows for robust estimates of the curvature and other geometric 
properties even when point-clouds have artifacts, such as
outliers as shown below. 

![
**Training with Noise.**  We show how Geometric Neural Operators (GNPs) for geometric estimation can be
trained for point-cloud representations to compensate for noise and other
artifacts.  ](fig/point_cloud_outliers.png){ width=100% }




## Usage

### GNP Models
The Geometric Neural Operator (GNP) models can be found in the class ``gnp.models``.
This includes the ``PatchGNP`` and the ``GNP`` classes. The ``GNP`` model class
allows for convolution types ``GraphConvolution``,
``BlockFactorizedConvolution``, and ``SeparableConvolution``.  For example,
instantiating a ``GNP`` model with the ``GraphConvolution`` would be
accomplished by 

```python
from gnp.models import GNP
full_model = GNP(
    node_dim=3,
    edge_dim=6,
    out_dim=1,
    layers=[64] * 10,
    conv_name="GraphConvolution",
    conv_args={"neurons": 128},
    nonlinearity="ReLU",
    skip_connection=True,
    device="cuda"
)

```
The ``GNP`` model requires as input the node dimensions, edge dimensions, and output dimensions, 
and the widths of each layer. The ``GNP`` model also requires as input the convolution 
type and any additional required arguments for the convolution type. In addition,
the ``GNP`` model requires the desired nonlinear activation from ``torch.nn`` and the 
device. For additional examples instantiating the ``GNP`` with other convolution 
types, see the package directory `geo_neural_op/examples`.

### Transferable Pre-trained Models: Usage

We provide example tutorials using our pre-trained ``PatchGNP`` for (i) computing 
geometric quantities, (ii) solving a Laplace-Beltrami PDE, and (iii) simulating geometric 
flows driven by mean curvature. Each of these 
examples utilize our core `GeometryEstimator` class that performs inference on 
the input data. For example, given two PyTorch tensors `pcd` and 
`orientation` of shape `[N, 3]`, one can instantiate the estimator using

```python
from gnp import GeometryEstimator
estimator = GeometryEstimator(pcd=pcd, 
                              orientation=orientation)
```
Additional arguments also can be passed to the estimator. For more details, see the 
[documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html).

### Estimating Geometric Quantities

To estimate the geometric quantities from the point-cloud one calls the function
`estimator.estimate_quantities` and passes a list of strings of the names of the 
desired geometric quantities. For example, this can be performed by 

```python
quantity_names = ["local_coordinates", "normals", "gaussian_curvature"]
quantities = estimator.estimate_quantities(quantity_names)
```

![
**Curvature Estimation.**  Estimated quantities of mean curvature (left) and Gaussian curvature (right) using
  ``GeometryEstimator.estimate_quantities``.
](fig/toroidal_curvatures.png){ width=90% }

### Shape Deformation Via Mean Curvature Flow
The package also has capabilities for performing shape deformations of point-clouds.
For example, shape evolution under mean-curvature driven flows can be simulated 
using the `mean_flow` function of the `GeometryEstimator` class

```python
flow_data = estimator.mean_flow(num_steps=1600, 
                                save_data_per_step=200,
                                delta_t=0.0002,
                                subsample_radius=0.005,
                                smooth_radius=0.06,
                                smooth_x=False)
```

![
**Mean Curvature Flow.**  Shape deformation via mean curvature flow using the pre-trained model in 
the ``GeometryEstimator`` class.
](fig/mean_curv_flow.png){ width=99%}

### Computing Stiffness Matrices using Generalized Moving Least Squares

The GNPs and other data-driven estimators can also be used within numerical
methods for solving PDEs and other problems. The matrices 
for the linear systems for the equations can be 
assembled for the point-cloud. These can be used in hybrid PDE
solvers that obtain solutions by using geometric estimates from 
GNPs while solving additional aspects of the problem using other 
numerical methods.  For example, the stiffness matrices for 
geometric Laplace-Beltrami PDEs can be computed using hybrid 
Generalized Moving Least Squares (GMLS) methods.  Here, the geometric contributions 
are obtained from the pre-trained PatchGNP model by calling
```python
stiffness, collocation_mask, outlier_mask = estimator.stiffness_matrix_gmls(
    drop_ratio=0.1
)
```
The ``drop_ratio`` parameter controls the amount in which the resulting linear 
system is overdetermined. 
For additional information on the `GeometryEstimator` class and ways its methods
can be used, see 
the [documentation](https://web.math.ucsb.edu/~atzberg/geo_neural_op_docs/html/index.html)
and [examples](https://github.com/atzberg/geo_neural_op/tree/main/examples).

# Conclusion

Our package [`GeoNeuralOp`](https://github.com/atzberg/geo_neural_op) 
provides methods for working with Geometric Neural Operators (GNPs).
This includes capabilities for (i) operator learning, 
(ii) geometric estimation from point-clouds, 
and (iii) other data processing tasks on domains
of general shape and topology.  We discussed here briefly a few of the 
available methods and models. The package also includes weights for 
transferable pre-trained basal foundation models for geometric estimation. 
For further updates, examples, and additional information please see 
[https://github.com/atzberg/geo_neural_op](https://github.com/atzberg/geo_neural_op) and 
[https://web.atzberger.org](https://web.atzberger.org).

# Acknowledgements

Authors research supported by NSF Grants DMS-1616353, DMS-2306101. Authors also
would like to acknowledge computational resources and administrative support at
the UCSB Center for Scientific Computing (CSC) with Grants NSF-CNS-1725797,
MRSEC: NSF-DMR-2308708, Pod-GPUs: OAC-1925717, and support from the California
NanoSystems Institute (CNSI) at UCSB. P.J.A. also would like to acknowledge a
hardware grant from Nvidia. The authors report there are no conflicts of interest. 

# Research impact statement

This software package is related to research work that has resulted 
in recent journal publication of two papers [@quackenbush_atzberger_gnps_2024],
[@quackenbush_atzberger_gnp_transfer_2025] and in submission of an additional
paper [@quackenbush_atzberger_extension_2026].  The package 
was recently posted 
on GitHub and has already been forked 5+ times and received 
30+ stars. This work also has resulted in invited conference 
and workshop talks, collaborations, and additional on-going papers 
and projects in active development. 

# AI usage disclosure

The authors also acknowledge the usage of generative AI in this repository
primarily for documentation and unit tests. The AI-generated material also
have been manually edited and reviewed.

# References
