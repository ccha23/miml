---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3.8 (XPython)
  language: python
  name: xpython
---

# Mutual Information in Machine Learning

+++

Mutual information is a fundamental quantity in information theory. It is widely used in machine learning to measure statistical dependency among different features in data. Applications are numerous, ranging from classification, clustering, representation learning, and other tasks that require the selection/extraction of lower-dimensional features of the data without losing valuable information. Although mutual information has a precise formula defined in terms of a probability model, it must be estimated for real-world data with an unknown probability model.

+++

In this lecture series, we will dive into some of the applications and estimations of mutual information in machine learning. Registered participants have hands-on coding experience using the virtual teaching and learning environment DIVE offered by CityU CS Department.

+++

## How to run the notebooks

+++

To run the notebooks with temporarily without any setup, click 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ccha23/mimldive/HEAD?urlpath=git-pull?repo%3Dhttps%3A%2F%2Fgithub.com%2Fccha23%2Fcscit21%26urlpath%3Dlab%2F%2Ftree%2Fcscit21).

+++

````{caution}

The binder service is convenient but the storage is temporary and it can take some times to build and launch the server. You also have limited computing resources to run the notebook.

````

+++

To run the notebook with persistent storage, install [docker](
https://docs.docker.com/get-started/#download-and-install-docker) on your computer and follow one of the following methods.

+++

## To run the notebooks in JupyterLab

+++

**Step 1** Run the docker in a terminal from a working directory of your choice.

+++

For Mac/Linux shells:  
```markdown
docker run --rm -p 10000:8888 -m 4g \
        -v "${PWD}":/home/jovyan \
        chungc/mimldive:v0.2 \
        start-notebook.sh --NotebookApp.token=''
```

+++

For Windows PowerShell:  
```markdown
docker run --rm -p 10000:8888 -m 4g `
        -v ${PWD}:/home/jovyan `
        chungc/mimldive:v0.2 `
        start-notebook.sh --NotebookApp.token=''  
```

+++

````{note}

- It may take a couple minutes to run for the first time as it needs to download the docker image. Subsequent run should be fast.
- Port 10000 specified by `-p` should be free for use. Otherwise, change it to a free port on your computer.
- The maximum memory limit is set to be 4GB by `-m`. You should set an appropriate value according to the memory available on your computer.
    
````

+++

**Step 2** Pull the notebooks in a web browser by visiting [this link][gp] or from the JupyterHub launcher button.  

[gp]: http://localhost:10000/git-pull?repo=https%3A%2F%2Fgithub.com%2Fccha23%2Fcscit21&urlpath=lab%2Ftree%2Fcscit21&branch=main

+++

````{tip}

- You can work on the notebooks under the `cscit21` subfolder. Clicking the above link again will automatically pull and merge changes from the repo, without overwritting your changes.
- To finish, stop the notebook server by pressing `Control-C` in the terminal that runs the docker or close the terminal/PowerShell.
- To restart, run the docker command again from the same working directory.

````

+++

## To run the notebook in VS Code

+++

**Step 1** Install [Visual Studio Code (VS Code)](https://code.visualstudio.com/) and the extension [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

+++

**Step 2** Run VS Code  
- click `View`->`Command Palette` 
- Enter `Remote-Containers: Clone Repository in Container Volume...`. There is also a command for using a Named Container Volume instead.
- Enter the repository url `https://github.com/ccha23/mimldive.git`

+++

You can now work on the notebooks in the `cscit21` subfolder and your files will be kept in a persistent docker volume.

+++

````{tip}

You can also start the jupyter lab server and open it in a browser by opening a terminal (`` Control-Shift-` ``) and run
  ```
  jupyter lab --NotebookApp.token=''
  ```

For more details, see the [vscode-remote-try-python](https://github.com/microsoft/vscode-remote-try-python) repository.

````
