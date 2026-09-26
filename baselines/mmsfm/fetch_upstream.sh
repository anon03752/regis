#!/bin/bash
# Download the pinned MMSFM sources and apply the Ising/MNIST patch.
set -e
cd "$(dirname "$0")"

MMSFM_COMMIT=f8c702b15fba75677d3a89fed2190c6ca9f3e106       # Shakeri-Lab/MMSFM (ICML 2025)
CFM_COMMIT=af8fec6f6dc3a0dc7f8fb25d2ee0ca819fa5412f         # atong01/conditional-flow-matching, upstream's own pin (make_venv.sh)

fetch() {  # <github project> <commit> <outname>
    wget -q -O "$3.zip" "https://github.com/$1/archive/$2.zip"
    unzip -q "$3.zip"
    mv "$3-$2" "$3"
    rm "$3.zip"
}

mkdir -p upstream
cd upstream
PATCH_SUM=$(sha256sum ../patches/mmsfm_ising.patch | cut -d' ' -f1)
if [ -f MMSFM/.regis_patched ] && [ "$(cat MMSFM/.regis_patched)" != "$PATCH_SUM" ]; then
    echo "the checkout carries another version of the patch; fetching it afresh"
    rm -rf MMSFM
fi
if [ ! -d MMSFM ]; then
    echo "downloading Shakeri-Lab/MMSFM @ ${MMSFM_COMMIT}"
    fetch Shakeri-Lab/MMSFM "$MMSFM_COMMIT" MMSFM
fi
if [ ! -d MMSFM/conditional-flow-matching ]; then
    echo "downloading atong01/conditional-flow-matching @ ${CFM_COMMIT}"
    (cd MMSFM && fetch atong01/conditional-flow-matching "$CFM_COMMIT" conditional-flow-matching)
fi
if [ ! -f MMSFM/.regis_patched ]; then
    echo "applying patches/mmsfm_ising.patch"
    patch -p1 -d MMSFM < ../patches/mmsfm_ising.patch
    echo "$PATCH_SUM" > MMSFM/.regis_patched
fi
echo "done. Extra dependencies (nothing is installed from the checkout):"
echo "    pip install torchsde==0.2.6 POT joblib tqdm wandb"
