#!/bin/bash

# Define absolute paths to avoid directory jumping confusion
BASE_DIR=$(pwd)
INPUT_DIR="$BASE_DIR/internship/source_codes"
OUTPUT_DIR="$BASE_DIR/internship/assembly_codes"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Silence Wine's annoying "fixme:" warnings to keep your terminal clean
export WINEDEBUG=-all

# Count total files for a progress tracker
total=$(ls -1 "$INPUT_DIR"/*.c 2>/dev/null | wc -l)
count=1

echo "Found $total .c files. Starting compilation..."

# Move into the input directory so Borland doesn't struggle with Linux paths
cd "$INPUT_DIR" || exit

# Loop through every .c file
for file in *.c; do
    echo "[$count/$total] Compiling $file..."

    # Run the Borland compiler via Wine
    wine /opt/borlandc/BIN/bcc32.exe -S -c -x-RT -D__CODEGUARD__ "$file" > /dev/null 2>&1

    # Extract the filename without the .c extension
    basename="${file%.c}"

    # Check if the compiler successfully generated the .asm file, then move it
    if [ -f "${basename}.asm" ]; then
        mv "${basename}.asm" "$OUTPUT_DIR/"
    else
        echo "  -> Warning: Failed to generate assembly for $file"
    fi

    count=$((count + 1))
done

echo "All done! Check the '$OUTPUT_DIR' folder for your files."