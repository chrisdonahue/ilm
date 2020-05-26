DATA_DIR=raw_data/roc_stories
rm -rf ${DATA_DIR}
mkdir -p ${DATA_DIR}
pushd ${DATA_DIR}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=15GLH9Kg-U0QANhEOwvgmsXycAO6OPFA7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=15GLH9Kg-U0QANhEOwvgmsXycAO6OPFA7" -O roc_stories_split.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz roc_stories_split.tar.gz
rm *.tar.gz
sha256sum *.txt
echo "If you use this dataset, please cite https://arxiv.org/pdf/1604.01696.pdf (Mostafazadeh et al. 2016)"
popd
