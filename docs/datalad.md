# DataLad

The easiest way to access the ACNets datasets is via DataLad, which is a version control tool for data, built on top of git. DataLad provides a handy command line tool to download data, share it, and track changes.

DataLad offers several features, but they are beyond the scope of this document. Here are some basic functionalities that we are interested in:

1. Cloning a remote data repository can be done using `datalad clone <repository>`. This does not download the whole dataset, but only creates folder structures and links to the files.

2. To download a file or folder, first change the current path to the cloned directory, and then use `datalad get <filename>`.

3. And to remove a downloaded file use `datalad drop <filename>`. Note that only downloaded file will be removed; it doesn't remove the file from the remote repository.

4. If you've changed the data, use `datalad save -r -m '<message>'` to save the latest state of the dataset and store changes.

5. To send your saved changes to the server, use `datalad push --to origin`.

6. And to export the whole dataset to an archive file, use `datalad export-archive -t tar -c bz2 ../dataset_vXXXX.YY.tar.bz2 --missing-content ignore`. Depending on the size of the dataset it takes a while to create the archive. Upon finishing the export, you can then backup or share the exported archive.


## Access HPC datasets (using DataLad)

The following code uses DataLad to clone available datasets from HPC into your local machine:

```bash
cd data/

datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~julia2018
datalad clone ria+ssh://iris-cluster:/work/projects/acnets/backup/datalad_riastore#~dosenbach2007
```

This will create folder structures and links for the two datasets, respectively in `data/julia2018/` and `data/dosenbach2007/` folders, but leave the content of some files on the remote server.

Most small files such as `.tsv` or `.md` texts will be available locally right away, but larger files like Nifti images remain inaccessible until you explicitly download them with DataLad.

The following code, for example, shows how to download all the scans for a given subject (sub-NVGP01):

```bash
cd data/julia2018/

datalad get bids/sub-NVGP01
```

Upon successful download, the content of all the files inside the `data/julia2018/bids/sub-NVGP01/` will be available on your local machine.


## FAQ
<details>
<summary><b>How can I edit the linked files?</b></summary>

Notice that even those downloaded contents cannot be directly edited because they are linked and managed by DataLad. In case you want to edit those linked files, you have to first *unlock* them:

```bash
datalad unlock derivatives/fmriprep/dataset_description.json
```

Now you can edit the content of `dataset_description.json` and perhaps save your changes and push them to the remote repository with `datalad save` and `datalad push --to origin` commands.
</details>
