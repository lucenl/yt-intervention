BASE_DIR="/media/data/lucen/codebase/yt-sock-puppet/output/profiles"
FILE="experiment_stats_failed.txt"

if [[ ! -f "$FILE" ]]; then
  echo "Error: File '$FILE' not found."
  exit 1
fi

dirs=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  path="$BASE_DIR/$line"
  [[ -d "$path" ]] && dirs+=("$path")
done < "$FILE"

count=${#dirs[@]}
if (( count == 0 )); then
  echo "No matching directories found under $BASE_DIR."
  exit 0
fi

echo "Found $count director$([[ $count -gt 1 ]] && echo "ies" || echo "y") under $BASE_DIR:"
printf '  %s\n' "${dirs[@]}"

read -p "Proceed with removal? [y/N] " ans
if [[ "$ans" =~ ^[Yy]$ ]]; then
  sudo rm -rf "${dirs[@]}"
  echo "Removed $count director$([[ $count -gt 1 ]] && echo "ies" || echo "y")."
else
  echo "No folders were removed."
fi
