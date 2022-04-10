# Running notebooks on your local machine

**:warning: please ignore this document if you already have your IDE ready.**


You need Jupyter or Colab to run the notebooks locally, but before that make sure you have all the requirements installed and the `julia2018` dataset is properly downloaded into the `data/` directory. Here is the step-by-step instruction:

1. Large datasets are stored on UNI HPC. First, you need to configure your SSH to access HPC via `iris-cluster` address by adding the following configs to the `~/.ssh/config` file. Replace <your_hpc_username> with your actual HPC username.

```ssh-config
Host iris-cluster
  Hostname access-iris.uni.lu
  User <your_hpc_username>
  Port 8022
  ForwardAgent no
```

You can verify it by running `ssh iris-cluster` in your Terminal.

2. Use [DataLad](https://www.datalad.org/get_datalad.html) to download datasets from the HPC.

```bash
cd data/

# julia2018 dataset
datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~julia2018

# dosenbach2007 dataset
datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~dosenbach2007
```

3. Next, you need to install the required development packages:

```bash
conda env create --file environment.yml
conda activate acnets
```

1. (optional) And finally, run Jupyter and locate the `notebooks/` directory:

```bash
jupyter notebook
```

Now you are able to either use the Jupyter UI or connect Google Colab to your local machine. VSCode is however preferred as some configurations will be automatically set.


## FAQ

<details>

<summary>
<b>What is the <code>project_dir</code> variable in some notebooks?</b>
</summary>

It is used to access datasets and store outputs in appropriate places.

`project_dir` is a relative path from the current working directory (i.e., `pwd`) referring to the project root folder. The reason for this variable is that different tools run notebooks in different working directories: some run the notebooks in the project directory, and some other run them in the notebook folder.


You can get rid of this relative variable by configuring jupyter to start in the project root. For example, add the following config to your Visual Studio Code workspace settings (`.vscode/settings.json` file in your workspace):

```json
"jupyter.notebookFileRoot": "${workspaceFolder}"
```

Then it's always safe to remove `project_dir` or set it to `'.'` in any notebooks.
</details>
