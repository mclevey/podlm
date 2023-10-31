# `podlm` Probabilistic Opinion Dynamics with Language Models

# Overview

# Preparing Analyses

To start a new analysis, initialize an analysis directory with `curry init`. For example, to initialize an analysis directory called `mclevey_20231031_rebuilding_pipeline`, we can supply the keywords "rebuilding pipeline":

```bash
curry init "rebuilding pipeline"
```

Or, equivalently:

```bash
curry init rebuilding_pipeline
```
 
The next step is to edit the analysis config file in the root of your analysis directory, e.g., `analyses/mclevey_20231031_rebuilding_pipeline/config.yaml`. Once you've finished with `config.yaml`, you can run the pipeline by supplying `curry run` with the path to the analysis directory. 

```bash
curry run analyses/mclevey_20231031_rebuilding_pipeline
```

It will take some time for the pipeline to run. When it finsihes, the contents of `pipeline/_export_/` will be filed in `analyses/<ANALYSIS_DIR_NAME>/_import_/results`.

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



