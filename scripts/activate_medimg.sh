#!/bin/bash

export PROJECT_HOME=/vol/biomedic2/bglocker_studproj/zc125
export TMPDIR=$PROJECT_HOME/tmp

if [ -f "$PROJECT_HOME/software/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$PROJECT_HOME/software/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate "$PROJECT_HOME/envs/medimg"

cd "$PROJECT_HOME"
echo "Activated medimg environment:"
which python
python --version
