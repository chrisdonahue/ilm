DATA_DIR=raw_data/arxiv_cs_abstracts
rm -rf ${DATA_DIR}
mkdir -p ${DATA_DIR}
pushd ${DATA_DIR}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1N3MbvpgZAmNgiZgnpXAQFzHrU7Tt3Blb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1N3MbvpgZAmNgiZgnpXAQFzHrU7Tt3Blb" -O arxiv_cs_abstracts.txt.gz && rm -rf /tmp/cookies.txt
gunzip arxiv_cs_abstracts.txt.gz
sha256sum arxiv_cs_abstracts.txt
popd
