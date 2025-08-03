set -e

ROOT=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)

FROM="$HOME/models"
TO="$ROOT/models"

cd "$FROM"
echo "Creating/replacing symlinks"

while IFS= read -r rel; do (
  abs="$(realpath "$rel")"

  cd "$TO"
  mkdir -p "$(dirname "$rel")"
  ln -fs "$abs" "$rel"

) done < <(find -mindepth 2 -maxdepth 2 -type f)

echo "Removing broken symlinks"
(cd "$TO" && find -L -type l -exec rm {} \;)

echo "Done"


