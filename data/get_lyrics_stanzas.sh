DATA_DIR=raw_data/lyrics_stanzas
rm -rf ${DATA_DIR}
mkdir -p ${DATA_DIR}
pushd ${DATA_DIR}
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1y46IMOa_oB9K-uD8gVmsGLcz6RQCPD9_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1y46IMOa_oB9K-uD8gVmsGLcz6RQCPD9_" -O lyrics_stanzas_split.tar.gz && rm -rf /tmp/cookies.txt
tar xvfz lyrics_stanzas_split.tar.gz
rm *.tar.gz
sha256sum *.txt
popd
