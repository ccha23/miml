#!/bin/bash

for cmd in "./preprocess.sh" "cp miml.ipynb ../release/" "rm -rf ~/www/miml/* && jupyter-book build --path-output ~/www/miml ." "./postprocess.sh" "scp -r ~/www/miml/_build/html/* ccha23@gateway:~/www/miml/"
do
    read -r -p "${cmd}?[Y/n] " input

    case $input in
        [yY][eE][sS]|[yY]|'')
    echo "Executing..."
    eval $cmd
    ;;
        [nN][oO]|[nN])
    echo "Skipped..."
        ;;
        *)
    echo "Invalid input..."
    exit 1
    ;;
    esac
done