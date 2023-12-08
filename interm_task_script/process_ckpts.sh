process_json() {
  local json_file=$1 # The JSON file to process

  # Check if the file exists
  if [[ ! -f "$json_file" ]]; then
    echo "JSON file not found: $json_file"
    return 1
  fi

  # Loop through each key-value pair in the JSON file
  jq -c 'to_entries[]' "$json_file" | while read -r entry; do
    local key=$(echo $entry | jq -r '.key')
    local value=$(echo $entry | jq -r '.value')
    # Call the user-provided callback function with key and value
    $2 "$key" "$value"
  done
}

