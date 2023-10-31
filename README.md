# `podlm` Probabilistic Opinion Dynamics with Language Models

# Overview

# Running Analyses

```bash
curry init "rebuilding pipeline"
```

Edit `analyses/mclevey_20231031_rebuilding_pipeline/config.yaml`.

```bash
curry run "rebuilding pipeline"
```

> [!info] Note that `curry run rebuilding_pipeline` will also work.

## Config Files

Anytime you run the pipeline, you will need to supply two config files:

1. The config file in your analysis directory, e.g. `analyses/mclevey_20231031_rebuilding_pipeline/config.yaml`, and
2. The `private.yaml` config file in the root directory of the project, which contains sensitive information like credentials and does **not** get committed to the repository.


# `pdpp` Pipeline

```bash
cd pipeline && pdpp graph
```

![](pipeline/dependencies_sparse.png)

# Environments

# Packages

## `curry`

## `podlm`



