set -e

# Kill children when done (yikes):
trap "kill -- -$$" EXIT

# Go to our parent directory:
ROOT=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
cd "$ROOT"

# Enter virtualenv:
source venv/bin/activate

# Start ComfyUI with CPU fallback for methods not supported by MPS:
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec -a comfyui python main.py --highvram --force-upcast-attention --preview-method auto "$@" &
PID="$!"

# Start sigkey to control the server: 
sigkey -2k shift_r "$PID" \
  --after-stop 'afplay /System/Library/Sounds/Bottle.aiff  -t 0.1' \
  --after-cont 'afplay /System/Library/Sounds/Funk.aiff -t 0.1' \
  --quiet 

# Clean up:
kill "$PID"
