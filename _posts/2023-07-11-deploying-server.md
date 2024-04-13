---
layout: post
title: Deploying a Server for Bioinformatics Research
created_date: 2023-07-11
last_updated_date: 2024-04-14
description: how to deploy a server for bioinformatics research
tags: deployment server Ubuntu
categories: computer
featured: true
toc:
  sidebar: left
---

Recently, our lab acquired a server equipped with a standard Ubuntu operating system from `Inspur`, and I am tasked with configuring it to fulfill the specific requirements of our bioinformatics research. Given that my expertise in Linux is limited, I dedicated several days to this endeavor, and eventually completed the deployment process. The purpose of this guide is to assist researchers facing similar demands in comprehending the steps to configure their servers. Additionally, it aims to address potential issues that may arise during the configuration process, along with their respective solutions.

## Create a new user (with root privileges)

Typically, the server comes with a default user named after the vendor, in my instance, `inspur`. This user is a regular user but can gain root privileges by executing commands using the `sudo` prefix followed by typing its password. To create a custom account with similar privileges, follow these steps:

```bash
sudo useradd -d "/home/<user_name>" -m -s "/bin/bash" <user_name>
```

- `-d "/volume1/home/<user_name>"` will set `/volume1/home/<user_name>` as home directory of the new Ubuntu account.
- `-m` will create the user’s home directory.
- `-s "/bin/bash"`will set `/bin/bash` as login shell of the new account.
  This command will create a regular account `<user_name>`. If you want `<user_name>` to have root privileges, type:

```bash
sudo useradd -d "/home/<user_name>" -m -s "/bin/bash" -G sudo <user_name>
```

- `-G sudo` ensures `<user_name>` to have admin access to the system.

To set the password of the new account, conduct:

```bash
sudo passwd <user_name>
```

After running the command, you will be prompted to type your password. Please note that Ubuntu will not display the password you are typing explicitly or implicitly (like dots). Just type the password you want to set and press `Enter`.

## Change terminal prompt (optional)

A beautiful terminal prompt can bring a beautiful day. To change the terminal prompt, conduct

```bash
cd ~
vim .bashrc
```

You will see a paragraph like this:

```text
# uncomment for a colored prompt, if the terminal has the capability; turned
# off by default to not distract the user: the focus in a terminal window
# should be on the output of commands, not on the prompt
#force_color_prompt=yes
```

Uncomment the last line `#force_color_prompt=yes`. Below this paragraph you will also see some codes:

```bash
if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
```

Modify the first `PS1`:

```bash
PS1='\[\033[35m\]\t\[\033[m\]-\[\033[36m\]\u\[\033[m\]@\[\033[32m\]\h:\[\033[33;1m\]\w\[\033[m\]\$ '
```

This is my `PS1` value. Save the `.bashrc` file, close your current terminal and open a new one. The terminal prompt will look like this:

<span style="color:magenta">23:02:02</span>-<span style="color:skyblue">tdeng</span>@<span style="color:green">inspur-NP5570M5:</span><span style="color:gold">~/data</span>$ .

## Enable remote access

If you wish to access the server from outside its physical location, you need to enable remote access. In my case, I connected the server to the campus network, allowing me to access it from any location within the campus. To enable remote access, you need to install the `openssh-server`:

```bash
sudo apt update
sudo apt install openssh-server
```

If the firewall `UFW` is enabled, make sure to open the SSH port:

```bash
sudo ufw allow ssh
```

To test whether you can access the server from a Windows system:

```bash
telnet <remote_ip> <remote_port>
```

[This website](https://linuxize.com/post/how-to-enable-ssh-on-ubuntu-20-04/) might be useful.

When you log in to the server using the newly created user with bash, you might encounter an error like this:

```bash
/usr/bin/xauth: file /home/<user_name>/.Xauthority does not exist
```

Solution:

```bash
chown <user_name>:<user_name> -R /home/<user_name>
```

## Connect to GitHub

I suppose you already have a GitHub account. Install `git` first:

```bash
sudo apt install git
git --version
```

Then configure certain information about your GitHub account:

```bash
git config --global user.name "<github_account_name>"
git config --global user.email "<github_account_email>"
```

Connect to GitHub:

```bash
ssh-keygen -C "<github_account_email>" -t rsa  # default: just press Enter 3 times
cd ~/.ssh
vim id_rsa.pub  # open the id_rsa.pub file
```

Finally, copy the text in `id_rsa.pub`, log in GitHub, and create an SSH key at `Settings` &rarr; `SSH and GPG keys` &rarr; `New SSH key`.

Test the connection:

```bash
ssh -T git@github.com
```

## Configure the python environment

### Install Anaconda

Just follow the [official installation guide](https://docs.anaconda.com/free/anaconda/install/linux/). I prefer install anaconda at `/usr/local/anaconda3`. You don't need to create this folder in advance. During the installation you will have chance to specify the installation directory.

To initialize conda, conduct

```bash
/path/to/conda init  # /usr/local/anaconda3/bin/conda in my case
```

and reopen the terminal.

### Create/delete environments

I recommend creating new environments and installing site packages with root privileges (`sudo su`) to restrict regular users from modifying the environments. If a regular user wants to update an environment, they should contact the system administrator for assistance. If he/she doesn't and conduct a command secretly like

```bash
conda update --all
```

he/she will proceed with the update plan but finally fail with error info:

```bash
Proceed ([y]/n)? y


Downloading and Extracting Packages

Preparing transaction: done
Verifying transaction: failed

EnvironmentNotWritableError: The current user does not have write permissions to the target environment.
  environment location: /usr/local/anaconda3/envs/bio
  uid: 1001
  gid: 1001
```

The commands for creating new environment are:

```bash
# create with a specified name
conda create --name <new_env_name> python=3.11 --no-default-packages
# create with a specified location
conda create --prefix /path/to/directory python=3.11 --no-default-packages
```

- `--name <new_env_name>` will set the name of the new environment.
- `--prefix /path/to/directory` will set the path to the directory where you want to create the environment。
- `python=3.11` means conda will install python 3.11 in the new environment.
- `--no-default-packages` will only install python. No other site packages will be included.

I did not modify the `base` environment and proceeded to create two new environments: `jupyter` and `bio`. `jupyter` only contains packages related to jupyterhub, while `bio` encompasses all the necessary packages for research purposes.

If you wish to delete an environment for any reason, utilize the following command:

```bash
conda remove --name <env_name> --all
```

### Install python packages

#### Mamba

Before install other python packages, I recommend to install `conda-libmamba-solver` first, which is a faster solver:

```bash
conda update -n <env_name> conda
conda install -n <env_name> conda-libmamba-solver
conda config --set solver libmamba
```

The plugin needs to be present in the same environment you use `conda` from; most of the time, this is your `base` environment.

#### JupyterHub

You may want to use `JupyterHub`.

```bash
conda install -c conda-forge jupyterhub jupyterlab notebook jupyter-lsp-python jupyterlab-lsp
```

I recommend to install the [jupyterlab-lsp](https://github.com/jupyter-lsp/jupyterlab-lsp), a powerful coding assistance for JupyterLab. Another useful plugin is [jupyterlab-execute-time](https://github.com/deshaw/jupyterlab-execute-time), which can display cell timings in JupyterLab. Use the following command to install it:

```bash
conda install -c conda-forge jupyterlab_execute_time
```

Refer to [this website](https://jupyterhub.readthedocs.io/en/stable/tutorial/getting-started/config-basics.html) for the configuration of JupyterHub.

Refer to
[this website](https://professorkazarinoff.github.io/jupyterhub-engr114/systemd/) for how to run JupyterHub as a system service.

Refer to
[this website](https://linuxconfig.org/how-to-start-service-on-boot-on-ubuntu-20-04) for how to start the service on boot.

#### Add/delete an environment as a kernel

To add an environment as a kernel:

```bash
conda activate <env_name>  # or /path/to/directory if you create the env with --prefix
conda install ipykernel  # if the env doesn't contain this package
python -m ipykernel install --name <kernel_name>
```

These commands add `<env_name>` environment as a kernel with name `<kernel_name>`. If your Python is 3.11, you may need to modify the last command:

```bash
python -Xfrozen_modules=off -m ipykernel install --name bio
```

To delete a kernel:

```bash
jupyter kernelspec list
jupyter kernelspec uninstall <kernel_name>
```

#### Other packages

Our research involves deep learning, so I need to install `pytorch` and `tensorflow` along with other required packages:

```bash
conda install -c pytorch -c nvidia -c conda-forge scvi-tools tensorflow torchvision torchaudio
conda install -c conda-forge scanpy squidpy ipykernel ipywidgets rpy2 opencv biopython
conda install -c conda-forge xgboost lightgbm catboost
```

Note: `pytorch` and `pytorch-lightning` are dependencies of `scvi-tools` so you don't need to install these two packages again.

Sometimes you may use `conda search <package_name>` to search for a package with a specific build number. To install a specific version/build of a certain packages, conduct:

```bash
conda install <package_name>=<version>=<build_string>
```

#### Check pytorch/tensorflow

If you are also a user of `pytorch` or `tensorflow` and you have one or more available GPU(s), you can execute the following codes to verify whether the GPU(s) can be recognized and utilized by the respective deep learning frameworks:

```python
import torch
import tensorflow as tf

# check pytorch and cuda in use
print(torch.cuda.version)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# check tensorflow
print(tf.config.list_physical_devices('GPU'))
```

Here I also provide a script to ensure that `pytorch` can use the GPU(s) to train and test neural networks:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


device = torch.device('cuda')
num_epochs = 50
batch_size = 512
learning_rate = 0.01

# define image preprocessing
transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()])

# download the CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='data/',
                                            train=False,
                                            transform=transforms.ToTensor())

# load data
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=4,            # number of subprocesses to use for data loading
                                           pin_memory=True,          # the data loader will copy Tensors into CUDA pinned memory before returning them
                                           prefetch_factor=4,        # number of batches loaded in advance by each worker
                                           persistent_workers=True,  # the data loader will not shutdown the worker processes after a dataset has been consumed once
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=4,
                                          pin_memory=True,
                                          prefetch_factor=4,
                                          persistent_workers=True,
                                          shuffle=False)

# 3x3 convolution kernel
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# define the residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# define the structure of ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
# model = nn.DataParallel(model)  # uncomment this line if you have multiple GPUs


# define loss function
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# function for the update of learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# train the ResNet
total_step = len(train_loader)
curr_lr = learning_rate
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # forward step
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report every 10 steps
        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # update learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
```

Note that if you have multiple GPUs, you need to uncomment the line below the code which creates the model:

```python
# model = nn.DataParallel(model)
```

You can use this command to monitor the GPU(s) during training:

```bash
watch -n 0.2 nvidia-smi
```

## Configure the R environment

### Install R

Refer to [this website](https://www.digitalocean.com/community/tutorials/how-to-install-r-on-ubuntu-22-04) for instructions on adding the external repository maintained by CRAN for `APT` and subsequently installing `R`.

### Install RStudio

Follow the [official installation guide](https://posit.co/download/rstudio-server/). This should be easier than installing `JupyterHub`.

### Install R packages

As an example, let's install one of the most famous R package in the field of single-cell genomics, [`Seurat`](https://satijalab.org/seurat/index.html). Before the installation, you need to install some system-level dependencies first:

```bash
sudo apt install cmake pandoc pandoc-citeproc libcurl4-openssl-dev libfontconfig1-dev libharfbuzz-dev libfribidi-dev libfreetype6-dev libpng-dev libtiff5-dev libjpeg-dev imagemagick libmagick++-dev libhdf5-dev libgsl-dev libssl-dev libblas-dev liblapack-dev
```

Then the process of installing `Seurat` should be very smooth:

```r
chooseCRANmirror(graphics=FALSE)
install.packages("devtools")
remotes::install_github("satijalab/seurat", "seurat5", quiet = TRUE)
```

If you intend to install an extremely large R package, you'd better set a longer timeout:

```r
options(timeout=999)
install.packages("<large_package>")
```

When running `devtools::install_github()`, you may encounter an error complaining that the API rate limit has been exceeded. The solution to this issue is to create a GitHub token.

```r
usethis::create_github_token()
```

Run this code and log in to your GitHub account. Click `Settings` &rarr; `Developer settings` &rarr; `Personal access token` &rarr; `Tokens (classic)` and generate a token. Run

```r
usethis::edit_r_environ()
```

to open the `.Renviron` file, in which type

```text
GITHUB_PAT=<your_personal_key>
```

and save the file. Quit the current R session and start a new session. The error should be solved.

## Synchronize data

Refer to [this website](https://www.digitalocean.com/community/tutorials/how-to-use-rsync-to-sync-local-and-remote-directories) for detailed instructions on how to synchronize data stored on another server.

The key command is

```bash
rsync -r /path/to/sync/ <username>@<remote_host>:<destination_directory>
```

which "pushes" all contents in `/path/to/sync/` from the system you are logging in to `<destination_directory>` in the target system.

If you are synchronizing a large file, you may want to monitor the process:

```bash
watch -n <time_interval> du -sh /path/to/large/file
```

## Install some basic fonts

By default, some basic fonts in Windows are not installed in Linux, such as `Arial` and `Times New Roman`. These fonts are commonly used in papers and websites, and having them installed can improve the display of figures that expect these fonts to be available. You can install them by:

```bash
sudo apt install msttcorefonts
rm -rf ~/.cache/matplotlib
```

The `msttcorefonts` package is a collection of TrueType fonts from Microsoft. The second command clears the matplotlib cache directory located in the hidden `.cache` directory in the user's home directory.

## Troubleshooting

### Driver/library version mismatch

When you run `nvidia-smi`, you may get

```bash
Failed to initialize NVML: Driver/library version mismatch
```

[This answer](https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch/45319156#45319156) from stackoverflow may help. Briefly you can either reboot or unload the `nvidia` module. However, if both the ways can't help, you need to reinstall the nvidia drivers:

```bash
sudo apt purge nvidia* libnvidia*
sudo ubuntu-drivers install
```

and then `sudo reboot` your server.

Now, your server should be well-suited for your bioinformatics research and you know what to do when things go wrong. Enjoy it!
