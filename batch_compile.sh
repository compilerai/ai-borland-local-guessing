#!/bin/bash

# --- CONFIGURATION ---
OLD_SERVER="trishanku@kiwi.cse.iitd.ac.in" 
REMOTE_WORKSPACE="~/borland_remote_compiler"
LOCAL_DATA_DIR="./data/source_codes"
LOCAL_ASM_DIR="./data/assembly_codes"

echo "Initiating Remote Compilation via $OLD_SERVER..."

# Ensure the local assembly directory exists before we try to download into it
mkdir -p "$LOCAL_ASM_DIR"

# 1. Create remote workspace
ssh "$OLD_SERVER" "mkdir -p $REMOTE_WORKSPACE"

# 2. Sync the C files over
echo "1/3: Teleporting .c files to the old server..."
rsync -avz "$LOCAL_DATA_DIR/"*.c "$OLD_SERVER:$REMOTE_WORKSPACE/"

# 3. Run YOUR robust batch_compile logic remotely via SSH Here-Doc
echo "2/3: Running robust Borland Compiler loop through Wine..."
ssh "$OLD_SERVER" "bash -s" << 'EOF'
    # --- THIS CODE RUNS ON THE OLD SERVER ---
    cd ~/borland_remote_compiler || exit
    
    # Silence Wine's annoying "fixme:" warnings
    export WINEDEBUG=-all

    total=$(ls -1 *.c 2>/dev/null | wc -l)
    count=1

    echo "Found $total .c files on remote server. Starting compilation..."

    for file in *.c; do
        # Print progress every 100 files to avoid spamming the SSH connection
        if [ $((count % 100)) -eq 0 ] || [ $count -eq 1 ]; then
            echo "[$count/$total] Compiling..."
        fi

        # Run the Borland compiler via Wine
        wine /opt/borlandc/BIN/bcc32.exe -S -c -x-RT -D__CODEGUARD__ "$file" > /dev/null 2>&1
        
        count=$((count + 1))
    done
    
    echo "Remote compilation finished!"
EOF

# 4. Bring the ASM files back (Routing specifically to the ASM folder)
echo "3/3: Retrieving .asm files back to current server..."
rsync -avz "$OLD_SERVER:$REMOTE_WORKSPACE/"*.asm "$LOCAL_ASM_DIR/"

echo "Done! All .asm files are now locally available in $LOCAL_ASM_DIR."