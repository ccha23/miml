#!/bin/bash

for f in "miml" "part1/Formulation" "part1/DonskerVaradhan" "part1/Training" "part1/f-Divergence" "part2/MI"
do
    read -r -p "process ${f}?[Y/n] " input

    case $input in
        [yY][eE][sS]|[yY]|'')
    echo "Executing..."
    #nbgrader generate_assignment --force --assignment="$(dirname "${f}")"
    jupytext --from ipynb --to md:myst -o "${f}.md" "${f}.ipynb"
    perl -0777 -i -pe 's/---\s*\*\*(Definition|Proposition|Lemma|Corollary|Proof)\*\*[^\S\r\n]*\(?([^()]*)\)?[^\S\r\n]*(\X*?)---/````{prf:\l$1} $2$3````/gm' "${f}.md"
    # perl -0777 -i -pe 's/---\s*\*\*(.*?)\*\*(.*)(\X*?)---/````{prf:\l$1}$2$3````/gm' "${f}.md"
    perl -0777 -i -pe 's/(\*\*Solution\*\*\X*?)((?:\+\+\+|```|\Z))/````{toggle}\n$1````\n\n$2/gm' "${f}.md"
    perl -0777 -i -pe 's/---\s*\*\*(.*?)\*\*(.*)(\X*?)---/````{\l$1}$2$3````/gm' "${f}.md"
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