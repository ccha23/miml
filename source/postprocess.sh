#!/bin/bash

find ~/www/miml/_build/html -name '*.html' -print0 | xargs -I{} -0 perl -pi -e 's{https://mybinder.org/v2/gh/ccha23/cscit21/.*lab/tree/(.*ipynb">)}{https://mybinder.org/v2/gh/ccha23/mimldive/HEAD?urlpath=git-pull?repo%3Dhttps%3A%2F%2Fgithub.com%2Fccha23%2Fcscit21%26urlpath%3Dlab%2Ftree%2Fcscit21%2F$1}g' "{}"

find ~/www/miml/_build/html -name '*.html' -print0 | xargs -I{} -0 perl -pi -e 's{http://localhost:10000/hub/user-redirect/}{http://localhost:10000/}g' "{}"