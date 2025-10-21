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
  - [Ray + LLM](#ray--llm)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Ray Data (Data Processing)](#ray-data-data-processing)
  - [Ray Train (Distributed Training)](#ray-train-distributed-training)
  - [Ray Tune (Hyperparameter Optimization)](#ray-tune-hyperparameter-optimization)
  - [Ray Serve (Model Serving)](#ray-serve-model-serving)
  - [Ray + JAX / TPU](#ray--jax--tpu)
  - [Ray + Database](#ray--database)
  - [Ray + X (Integration)](#ray--x-integration)
  - [Ray-Project](#ray-project)
  - [Distributed Computing](#distributed-computing)
  - [Ray AIR](#ray-air)
  - [Cloud Deployment](#cloud-deployment)
- [Videos](#videos)
- [Papers](#papers)
- [Tutorials and Blog Posts](#tutorials-and-blog-posts)
- [Books](#books)
- [Courses](#course)
- [Cheatsheet](#cheatsheet)
- [Community](#community)

<a name="libraries" />

## Libraries

<a name="new-libraries" />

### New Libraries

This section contains libraries that are well-made and useful, but have not necessarily been battle-tested by a large userbase yet.


<a name="models-and-projects" />

## Models and Projects

### Ray + LLM
- [veRL](https://github.com/volcengine/verl/tree/main) veRL: Volcano Engine Reinforcement Learning for LLM
- [FastChat](https://github.com/lm-sys/FastChat) Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90% ChatGPT Quality
- [LangChain-Ray](https://github.com/ray-project/langchain-ray) Examples on how to use LangChain and Ray
- [Aviary](https://github.com/ray-project/aviary) Ray Aviary - evaluate multiple LLMs easily
- [LLM-distributed-finetune](https://github.com/AdrianBZG/LLM-distributed-finetune) Finetuning Large Language Models Efficiently on a Distributed Cluster, Uses Ray AIR to orchestrate the training on multiple AWS GPU instances.
- [LLMPerf](https://github.com/ray-project/llmperf) - A library for validating and benchmarking LLMs (updated through 2024)

### Reinforcement Learning

- [slime](https://github.com/THUDM/slime) - A LLM post-training framework aiming at scaling RL.
- [muzero-general](https://github.com/werner-duvaud/muzero-general) - A commented and documented implementation of MuZero based on the Google DeepMind paper (Schrittwieser et al., Nov 2019) and the associated pseudocode.
- [rllib-torch-maddpg](https://github.com/Rohan138/rllib-torch-maddpg) - PyTorch implementation of MADDPG (Lowe et al.) in RLLib
- [MARLlib](https://github.com/Replicable-MARL/MARLlib) - a comprehensive Multi-Agent Reinforcement Learning algorithm library
- [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) - A vectorized differentiable simulator for Multi-Agent Reinforcement Learning benchmarking

### Ray Data (Data Processing)

- [RayDP](https://github.com/intel-analytics/oap-raydp) - Distributed data processing library on Ray by running Apache Spark on Ray. Seamlessly integrates with other Ray libraries for E2E data analytics and AI pipeline.
- [Google Cloud Platform Ray Preprocessing](https://github.com/GoogleCloudPlatform/accelerated-platforms) - Examples of Ray data preprocessing pipelines for model fine-tuning on GCP.

### Ray Train (Distributed Training)

- [Ray Train Examples](https://docs.ray.io/en/latest/train/train.html) - Official Ray Train documentation with PyTorch, TensorFlow, and Hugging Face Accelerate examples for distributed training.
- [MinIO with Ray Train](https://blog.min.io/distributed-training-with-ray-train-and-minio/) - Distributed training examples using Ray Train with MinIO object storage.

### Ray Tune (Hyperparameter Optimization)

- [Ultralytics YOLO11 with Ray Tune](https://docs.ultralytics.com/integrations/ray-tune/) - Efficient hyperparameter tuning for YOLO11 object detection models using Ray Tune.
- [Softlearning](https://github.com/rail-berkeley/softlearning) - Reinforcement learning framework for training maximum entropy policies, official implementation of Soft Actor-Critic algorithm using Ray Tune.
- [Flambe](https://github.com/asappresearch/flambe) - ML framework to accelerate research and its path to production, integrates with Ray Tune.

### Ray Serve (Model Serving)

- [LangChain Ray Serve](https://python.langchain.com/docs/integrations/providers/ray_serve/) - Deploy LangChain applications and OpenAI chains in production using Ray Serve.

### Ray + JAX / TPU

- [Swarm-jax](https://github.com/kingoflolz/swarm-jax) - Swarm training framework using Haiku + JAX + Ray for layer parallel transformer language models on unreliable, heterogeneous nodes
- [Alpa](https://github.com/alpa-projects/alpa) - Auto parallelization for large-scale neural networks using Jax, XLA, and Ray


### Ray + Database

- [Balsa](https://github.com/balsa-project/balsa#cluster-mode) Balsa is a learned SQL query optimizer. It tailor optimizes your SQL queries to find the best execution plans for your hardware and engine.
- [RaySQL](https://github.com/andygrove/ray-sql) Distributed SQL Query Engine in Python using Ray
- [Quokka](https://github.com/marsupialtail/quokka) Open source SQL engine in Python 

### Ray + X (integration)

- [Ray MCP Server](https://github.com/pradeepiyer/ray-mcp) – Bridge AI assistants to Ray clusters; manage clusters, submit jobs, and monitor resources through a Model Context Protocol interface.
- [prefect-ray](https://github.com/PrefectHQ/prefect-ray) Prefect integrations with Ray
- [xgboost_ray](https://github.com/ray-project/xgboost_ray) Distributed XGBoost on Ray
- [Ray-DeepSpeed-Inference](https://github.com/tchordia/ray-serve-deepspeed) Run deepspeed on ray serve

### Ray-Project
- [SkyPilot](https://github.com/skypilot-org/skypilot) a framework for easily running machine learning workloads on any cloud through a unified interface
- [Exoshuffle-CloudSort](https://github.com/exoshuffle/cloudsort) the winning entry of the [2022 CloudSort Benchmark](http://sortbenchmark.org/) in the Indy category.

### distributed computing
- [Fugue](https://github.com/fugue-project/fugue) a unified interface for distributed computing that lets users execute Python, pandas, and SQL code on Ray without rewrites.
- [Daft](https://github.com/Eventual-Inc/Daft) is a fast, Pythonic and scalable open-source dataframe library built for Python and Machine Learning workloads.
- [Flower](https://github.com/adap/flower)(flwr) is a framework for building federated learning systems. Uses Ray for scaling out experiments from desktop, single GPU rack, or multi-node GPU cluster.
- [Modin](https://github.com/modin-project/modin): Scale your pandas workflows by changing one line of code. Uses Ray for transparently scaling out to multiple nodes.
- [Volcano](https://github.com/volcano-sh/volcano) is a batch system built on Kubernetes. It provides a suite of mechanisms that are commonly required by many classes of batch & elastic workloads.

### Ray AIR
- [Ray on Azure ML](https://github.com/microsoft/ray-on-aml) Turning AML compute into Ray cluster

### Cloud Deployment
- [Ray on AWS](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/aws.html) - Official guide for launching Ray clusters on AWS with CloudWatch monitoring
- [Ray on GCP](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/gcp.html) - Official guide for launching Ray clusters on Google Cloud Platform
- [Ray on Azure](https://docs.ray.io/en/latest/cluster/vms/user-guides/launching-clusters/azure.html) - Official guide for launching Ray clusters on Microsoft Azure

### Misc
- [AutoGluon](https://github.com/awslabs/autogluon) AutoML for Image, Text, and Tabular Data
- [Aws-samples](https://github.com/aws-samples/aws-samples-for-ray) Ray on Amazon SageMaker/EC2/EKS/EMR
- [KubeRay](https://github.com/ray-project/kuberay) A toolkit to run Ray applications on Kubernetes
- [ray-educational-materials](https://github.com/ray-project/ray-educational-materials) This is suite of the hands-on training materials that shows how to scale CV, NLP, time-series forecasting workloads with Ray.
- [Metaflow-Ray](https://github.com/outerbounds/metaflow-ray) An extension for Metaflow that enables seamless integration with Ray 

<a name="videos" />

## Videos

### Anyscale Academy & Official Tutorials
- [Anyscale YouTube Channel](https://www.youtube.com/@AnyscaleInc) - Official YouTube channel with Ray tutorials, conference talks, and educational content
- [Anyscale Academy](https://github.com/anyscale/academy) - Ray tutorials from Anyscale with accompanying videos on YouTube
- [Ray Crash Course](https://www.anyscale.com/blog/video-and-code-for-anyscale-academy-ray-crash-course-may-27-2020) - Introductory online class with video on Anyscale YouTube
- [Reinforcement Learning with Ray RLlib](https://www.anyscale.com/blog/video-and-code-for-anyscale-academy-reinforcement-learning-with-ray-rllib-june-24-2020) - Complete tutorial with video

### Conference Talks
- [Ray Summit 2024](https://www.anyscale.com/ray-summit/2024) - Annual Ray conference with recorded sessions on YouTube (Sep 30 - Oct 2, 2024)
- [Ray Summit 2025](https://raysummit.anyscale.com/) - Upcoming conference (Nov 3-5, 2025, San Francisco)

<a name="papers" />

### RLlib
- [Deep reinforcement learning at Riot Games by Ben Kasper](https://youtu.be/r6ErUh5sjXQ) - reinforcement learning for game development in production



## Papers

This section contains papers focused on Ray (e.g. RAY-based library whitepapers, research on RAY, etc). Papers implemented in RAY are listed in the [Models/Projects](#projects) section.

<!--lint ignore awesome-list-item-->

### Foundational Papers
- [Ray: A Distributed Framework for Emerging AI Applications (OSDI 2018)](https://www.usenix.org/system/files/osdi18-moritz.pdf) - The foundational paper presenting Ray's unified interface for task-parallel and actor-based computations. Demonstrates scaling beyond 1.8 million tasks per second.
- [Ray on arXiv](https://arxiv.org/abs/1712.05889) - arXiv version of the foundational Ray paper

<!--lint enable awesome-list-item-->

<a name="tutorials-and-blog-posts" />

## Tutorials and Blog Posts

### 2024-2025
- [Ray: Your Gateway to Scalable AI and Machine Learning Applications](https://www.analyticsvidhya.com/blog/2025/03/ray/) - Analytics Vidhya (March 2025) - Comprehensive guide to Ray's architecture and capabilities with practical project implementation
- [RAY: A Powerful Distributed Computing Framework for ML/AI](https://blog.spheron.network/ray-a-powerful-distributed-computing-framework-for-machine-learning-and-ai) - Spheron Network (June 2024) - Covers Ray's capabilities for scaling models and distributed computing
- [The Modern AI Stack: Ray](https://medium.com/bitgrit-data-science-publication/the-modern-ai-stack-ray-44be004bd1c0) - Medium (September 2024) - How Ray fits into the modern AI infrastructure
- [Understanding Iterations in Ray RLlib](https://www.tecracer.com/blog/2024/02/understanding-iterations-in-ray-rllib.html) - tecRacer (February 2024) - Deep dive into RLlib's learning iterations
- [Ray Summit 2024: Breaking Through the AI Complexity Wall](https://www.anyscale.com/blog/ray-summit-2024-recap) - Anyscale (2024) - Highlights from Ray Summit 2024, orchestrating 1M+ clusters per month
- [How Ray Helps Power ChatGPT](https://thenewstack.io/how-ray-a-distributed-ai-framework-helps-power-chatgpt/) - The New Stack - How OpenAI uses Ray for ChatGPT training coordination

### Earlier Resources
- [Programming in Ray: Tips for first-time users](https://rise.cs.berkeley.edu/blog/ray-tips-for-first-time-users) - Berkeley RISE Lab
- [Hacker News Discussion](https://news.ycombinator.com/item?id=27730807) - Community discussion about Ray
- [Load PyTorch Models 340 Times Faster with Ray](https://medium.com/ibm-data-ai/how-to-load-pytorch-models-340-times-faster-with-ray-8be751a6944c) - IBM
- [Writing Your First Distributed Python Application with Ray](https://www.anyscale.com/blog/writing-your-first-distributed-python-application-with-ray) - Anyscale official tutorial

## books 

- [Learning Ray](https://github.com/maxpumperla/learning_ray) Learning Ray - Flexible Distributed Python for Machine Learning

## course

- [RL course](https://github.com/anyscale/rl-course) Applied Reinforcement Learning with RLlib
- [MLops course](https://github.com/anyscale/mlops-course) MLops course

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
