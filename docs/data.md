# [OBSOLETE] Data

The easiest way to access the ACNets datasets is via DataLad, which is a version control tool for data, built on top of git. DataLad provides a handy command line tool to download data, share it, and track changes.

DataLad offers several features, but they are beyond the scope of this document. Here are some basic functionalities that we are interested in:

1. A remote data repository can be cloned locally using `datalad clone <repository>`. This does not download the whole dataset, but only creates folder structures and links to the files.

2. To download a file or folder, first change the current path to the cloned directory, and then use `datalad get <path>`.

3. To remove a downloaded file use `datalad drop <path>`. Downloaded file will be only removed locally; it is not removed from the remote repository.

4. If you've changed the data, use `datalad save -r -m '<message>'` to save the latest state of the dataset and store changes.

5. To submit your saved changes to the server, use `datalad push --to origin`.

6. And to export the whole dataset to an archive file, use `datalad export-archive -t tar -c bz2 ../dataset_vXXXX.YY.tar.bz2 --missing-content ignore`. Depending on the size of the dataset it takes a while to create the archive. Upon finishing the export, you can backup or share the archive file.


## [OBSOLETE] Access Julia2018 dataset on HPC

The following code clones *Julia2018* dataset from UNI.LU HPC into your local machine. Before running the code, you need to install `datalad` and `git` on your local machine, and properly configure your ssh.

 First, you need to configure your SSH to access HPC via `iris-cluster` address by adding the following configs to the `~/.ssh/config` file. Replace <your_hpc_username> with your actual HPC username.

```ssh-config
Host iris-cluster
  Hostname access-iris.uni.lu
  User <your_hpc_username>
  Port 8022
  ForwardAgent no
```

And then, download the data:

```bash
cd data/

datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~julia2018

# OR dosenbach2007 dataset
# datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~dosenbach2007
```

Alternatively, you can clone the datasets from the xCIT external drive (Samsung X5):


```bash
datalad install file:///Volumes/xcit_x5/acnets/data/julia2018

# OR dosenbach2007 dataset
# datalad install file:///Volumes/xcit_x5/acnets/data/dosenbach2007
```


This will create folder structures and links respectively in the `data/julia2018/`, but leave the content of large files on the remote server.

Most small files such as `.tsv` or `.md` texts will be available locally right away, but larger files like Nifti images remain inaccessible until you explicitly download them with DataLad.

The following code, for example, shows how to download all the scans for a given subject (sub-NVGP01):

```bash
cd data/julia2018/

datalad get sub-NVGP01
datalad get derivatives/fmriprep_2020/sub-NVGP01
```

Upon successful download, the content of all the files inside the respective folders will be available on your local machine.


## [OBSOLETE] FAQ
<details>
<summary><b>How can I edit the linked files?</b></summary>

You may notice that downloaded contents cannot be directly edited because they are linked and managed by DataLad. In case you want to edit those linked files, you have to first *unlock* them. For example:

```bash
datalad unlock derivatives/fmriprep/dataset_description.json
```

Now you can edit the content of the `dataset_description.json` and perhaps save your changes and push them to the remote repository with `datalad save` and `datalad push --to origin` commands.
</details>

<details>
<summary><b>How can I download the preprocessed fmriprep derivatives from HPC Iris cluster?</b></summary>

Change your working folder to cloned `data/julia2018/` and use the following datalad command to download the derivatives.

```datalad get -s iris-ria-storage derivatives/fmriprep_2020```

</details>
