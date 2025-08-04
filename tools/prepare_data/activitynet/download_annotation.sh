DATA_DIR="/home/yi/data/opentad/data/activitynet-1.3/annotations"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "${DATA_DIR} does not exist. Creating"
    mkdir -p ${DATA_DIR}
fi

# download annotations for ActivityNet-1.3
gdown https://drive.google.com/drive/folders/1HpTc6FbYnm-s9tY4aZljjZnYnThICcNq -O $DATA_DIR --folder
