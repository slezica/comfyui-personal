set -e

ROOT=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)

# Enter virtualenv:
source venv/bin/activate

# Upgrade plugins:
while IFS= read -r dir; do
  cd "$ROOT/$dir"

  echo "> Pulling $dir"
  [[ -d .git ]] && git pull

  echo "> Updating $dir requirements (if any)"
  [[ -f requirements.txt ]] && pip install -r requirements.txt || true

done < <(ls -d custom_nodes/*/ | grep -v __)

echo "> Updating independent requirements"
pip install --upgrade onnxruntime insightface

# Update core:
echo "> Pulling ComfyUI"
git checkout master
git pull

echo "> Updating ComfyUI requirements"
pip install -r requirements.txt

# echo "> Force-installing specific dependency versions"
# pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 numpy==1.23.1
#
echo "> Done"
