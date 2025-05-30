<!-- markdownlint-disable MD033 -->

# Download and Setup Instructions

There are a few ways you can download the samples to run with the IBM Z
Accelerated for PyTorch container. You can clone this repository with an
installed [git](https://git-scm.com) client, download it as a zip, or download
individual files using the Github web interface. With a git client you can clone
this repository directly onto an IBM Z&reg; system or from inside the container
if your local policies allow Internet access.

You might already be familiar with these techniques. These instructions are the
same for any Github repository or docker container.

# Download Files

Choose one of the following options.

## Option 1. Clone the Entire Repository

This option will give you access to the samples, but will download additional
files (e.g this document) that you might not need. Running the `git pull`
command will download updates that are later made to the repository.

A git client must be installed on your workstation, on an IBM Z system or
installed in the IBM Z Accelerated for PyTorch container, and Internet access
to github.com.

From the [home page] for this repository you can click the green 'Code' drop
down button to show and copy the git clone uri for the repository. Add this uri
to a `git clone` command to clone the repository into your current directory.

![Github repository green code drop down button](github-green-code-button-dropdown.png)

If the `git clone` was performed on your workstation, copy the files to the
IBM&reg; Z system where you will run the container. Use scp, ftps or any file
transfer solution available in your installation.

## Option 2. Download the Entire Repository

This option will give you access to the samples, but will download additional
files you might not need. To get updates from the repository, repeat these
instructions.

From the [home page] for this repository you can click the green 'Code' drop
down button. Click the 'Download ZIP' button to download the repository as a zip
file.

Extract the zip file either on your workstation and copy the samples to the IBM
Z system where you will run the container. Use scp, ftps or any file transfer
solution available in your installation.

## Option 3. Download Sample Files Individually

This option will give you access to the samples and not download extraneous
files. However, each file must be downloaded individually. To get updates from
the repository, repeat these instructions.

Go to the sample file in the Github web interface and download it with one of
the following:

- Click the 'Raw' button to view the contents of the file only. Then either:
  - Right click and select 'Save Page As...' (might be different in your web
    browser) to save the sample to your workstation.
  - Select the entire contents of the file, copy it, and then past the content
    to a file editor on your workstation.
- Right click the 'Raw' button in the github web interface and select 'Save Link
  As..' (might be different in your web browser) to download the content as a
  file.

# Make the Samples Available to the Container

If you have used `git clone` inside of the IBM Z Accelerated for PyTorch
container, `cd` to the directory containing the sample files.

Otherwise, we assume the repository or just the samples are now copied onto the
IBM Z environment that hosts the IBM Z Accelerated for PyTorch container.

One way of making host files available inside of a container is a
[bind mount](https://docs.docker.com/storage/bind-mounts/). This shares the
files between the host and the container. Changes to the files made on the host
or in the container are available to both. To use a bind mount, specify the `-v`
parameter with the name of a host directory or file, and the name of the
directory or file as it should appear inside the container. This parameter must
be supplied when the container is first started with `docker run`.

```bash
# The README files in the samples will use this variable when
# referring to the container image.
IMAGE_ID=icr.io/ibmz-accelerated-for-pytorch:X.X.X

# Share samples with a bind mount. Adjust the paths if needed
# for the download directory you used.
docker run -it -v ./ibmz-accelerated-for-pytorch/samples:/samples:z $IMAGE_ID /bin/bash
```

Another way is to use `docker cp` to copy the files into the container. For this
technique, the container must already be running via `docker run`. You might
need to open a second shell to the host system, or use
<kbd>Ctrl</kbd><kbd>P</kbd><kbd>Ctrl</kbd><kbd>Q</kbd> to detach from the
running container to get back to the host shell.

```bash
# To find the generated ID or name of the running container use:
docker ps

# Adjust the paths if needed for the download directory you used.
docker cp ./ibmz-accelerated-for-pytorch/samples <container_id_or_name>:/samples

# If you detached from the running container, get back to it with.
docker attach <container_id_or_name>
```

# Training on CPU

Some of the samples, or your own PyTorch applications, will train models.
Training should be performed on the CPU.

If problems persist, this might indicate a problem with how the model is
constructed, or it might be a PyTorch issue.

# Running the Samples

See the README.md file included with the sample for additional setup
instructions and for the commands to run them.

Currently there are two samples available:

[MNIST](mnist)

[Credit Card Fraud](credit-card-fraud)

# Saving Models or Code changes

With the `docker run --rm` parameter (which is used by the sample commands
here), any changes you make inside of a container are lost when you exit the
container. This includes any downloaded data or models or code that was created.
A few techniques are available to persist the data and make it available for
usage outside the container, or with other containers.

1. Don't use the `--rm` flag. Upon exit, the container will persist and can be
   restarted and reused.

   - Note without `--rm` it is easy to unintentionally create a number of
     persisted containers that can fill up your available storage. Use
     `docker ps -a` to view persisted yet stopped containers. Use `docker rm` to
     remove any that are no longer needed.
   - Using `docker run --name {some-name}` whenever you create a new container
     instance can also help prevent this as only one container with a given name
     may exist at a time. You can then either delete the old container, or
     `docker start {some-name}` it and resume your work where you left off.

2. `docker cp` can target a running container and copy files and directories out
   of the container and into your host file system.
3. Use bind mounts with the `docker run -v` or `docker run --mount` parameter.
   This will mount a host directory into the container, and any existing data or
   code cna be used inside the container, and any data or code updated or
   created in the directory will be persisted in the host directory for later
   reuse.By default,

   - Note that under podman this can result in some permission and ownership
     issues. These issues are are easily fixed with
     `podman unshare chown -R root:root {dir}` on the host. This changes the
     owner back to your user id (as `root` in that command actually refers to
     your user id and primary group and not the host root user).

4. Use [volumes](https://docs.docker.com/storage/volumes/) with the
   `docker run --mount` parameter. Volumes isolate the data from the host and
   can be easily shared between containers.

   - Under podman, you can easily populate a volume from a tarball on the host,
     or export a volume to a tarball with the
     [`podman import`](https://docs.podman.io/en/latest/markdown/podman-volume-import.1.html)
     and
     [`podman export`](https://docs.podman.io/en/latest/markdown/podman-volume-export.1.html)
     commands, respectively.

Use the technique that works best for your application to persist and share your
data.

You may also save the changes you made in a container by using
[docker commit](https://docs.docker.com/engine/reference/commandline/commit/) to
save the running container as an image. The entire state of the container,
software installed, etc will be retained. But note that directories and files
made available to the container via bind mounts will not be saved as part of the
container image. However you can use `-v` to bind mount that data again when
running the new container image.

<!--
Data that Tensorflow downloads into the container will be stored under the
`.cache` directory in the home directory of the current user
(`/home/ibm-user/.cache`). Persisting this data with a bind mount or volume
will allow you to reuse that data and avoid downloading it again.
-->

<!-- reference links -->

[home page]: https://github.com/IBM/ibmz-accelerated-for-pytorch
