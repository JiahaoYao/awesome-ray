<!--lint ignore double-link-->
# Awesome RAY [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)[<img src="https://raw.githubusercontent.com/ray-project/ray/master/doc/source/images/ray_header_logo.png" alt="Ray Logo" align="right" height="100">](https://github.com/ray-project/ray/)

![ray_logo](https://user-images.githubusercontent.com/20907377/175792492-a085fc63-3b5f-4804-8215-e9c45cb284aa.gif)

<!--lint ignore double-link-->
[Ray](https://github.com/ray-project/ray/) makes it effortless to parallelize single machine code — go from a single CPU to multi-core, multi-GPU or multi-node with minimal code changes.
<!--lint enable double-link-->

This is a curated list of awesome RAY libraries, projects, and other resources. Contributions are welcome!

## Contents

- [Libraries](#libraries)
- [Models and Projects](#models-and-projects)
- [Videos](#videos)
- [Papers](#papers)
- [Tutorials and Blog Posts](#tutorials-and-blog-posts)
- [Community](#community)

<a name="libraries" />

## Libraries

<a name="new-libraries" />

### New Libraries

This section contains libraries that are well-made and useful, but have not necessarily been battle-tested by a large userbase yet.


<a name="models-and-projects" />

## Models and Projects

### Reinforcmenet Learning

- [muzero-general](https://github.com/werner-duvaud/muzero-general) - A commented and documented implementation of MuZero based on the Google DeepMind paper (Schrittwieser et al., Nov 2019) and the associated pseudocode.
- [rllib-torch-maddpg](https://github.com/Rohan138/rllib-torch-maddpg) - PyTorch implementation of MADDPG (Lowe et al.) in RLLib
- [MARLlib](https://github.com/Replicable-MARL/MARLlib) - a comprehensive Multi-Agent Reinforcement Learning algorithm library

### Ray + JAX / TPU

- [Swarm-jax](https://github.com/kingoflolz/swarm-jax) - Swarm training framework using Haiku + JAX + Ray for layer parallel transformer language models on unreliable, heterogeneous nodes
- [Alpa](https://github.com/alpa-projects/alpa) - Auto parallelization for large-scale neural networks using Jax, XLA, and Ray


### Ray + Database

- [Balsa](https://github.com/balsa-project/balsa#cluster-mode) Balsa is a learned SQL query optimizer. It tailor optimizes your SQL queries to find the best execution plans for your hardware and engine.
- [RaySQL](https://github.com/andygrove/ray-sql) Distributed SQL Query Engine in Python using Ray

### Ray + X (integration)

- [prefect-ray](https://github.com/PrefectHQ/prefect-ray) Prefect integrations with Ray
- [xgboost_ray](https://github.com/ray-project/xgboost_ray) Distributed XGBoost on Ray

### Ray-Project
- [SkyPilot](https://github.com/skypilot-org/skypilot) a framework for easily running machine learning workloads on any cloud through a unified interface
- [Exoshuffle-CloudSort](https://github.com/exoshuffle/cloudsort) the winning entry of the [2022 CloudSort Benchmark](http://sortbenchmark.org/) in the Indy category.

### distributed computing
- [Fugue](https://github.com/fugue-project/fugue) a unified interface for distributed computing that lets users execute Python, pandas, and SQL code on Ray without rewrites.

### Ray AIR
- [Ray on Azure ML](https://github.com/microsoft/ray-on-aml) Turning AML compute into Ray cluster

### Misc
- [AutoGluon](https://github.com/awslabs/autogluon) AutoML for Image, Text, and Tabular Data
- [Aws-samples](https://github.com/aws-samples/aws-samples-for-ray) Ray on Amazon SageMaker/EC2/EKS/EMR
- [KubeRay](https://github.com/ray-project/kuberay) A toolkit to run Ray applications on Kubernetes

<a name="videos" />

## Videos

<a name="papers" />

### rllib 
- [Deep reinforcement learning at Riot Games by Ben Kasper](https://youtu.be/r6ErUh5sjXQ) - reinforcement learning for game development in production



## Papers

This section contains papers focused on Ray (e.g. RAY-based library whitepapers, research on RAY, etc). Papers implemented in RAY are listed in the [Models/Projects](#projects) section.

<!--lint ignore awesome-list-item-->

<!--lint enable awesome-list-item-->

<a name="tutorials-and-blog-posts" />

## Tutorials and Blog Posts


- [Programming in Ray: Tips for first-time users](https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users)
- [Reddit post](https://news.ycombinator.com/item?id=27730807) 
- [Load PyTorch Models 340 Times Faster with Ray](https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-ray-8be751a6944c)

## books 

- [Learning Ray](https://github.com/maxpumperla/learning_ray) Learning Ray - Flexible Distributed Python for Machine Learning

## course

- [RL course](https://github.com/anyscale/rl-course) Applied Reinforcement Learning with RLlib


## cheatsheet

- [Ray design doc](https://docs.google.com/document/d/167rnnDFIVRhHhK4mznEIemOtj63IOhtIPvSYaPgI4Fg/edit#heading=h.5jeo2fupy3yv)

<a name="community" />

## Community

- [RAY Discussions](https://discuss.ray.io/)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/ray)
- [Slack](https://forms.gle/9TSdDYUgxYs8SA9e8)
- [GitHub Issues](https://github.com/ray-project/ray/issues)
- [Meetup Group](https://www.meetup.com/Bay-Area-Ray-Meetup/)
- [Twitter](https://twitter.com/raydistributed)


## Contributing

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
