rm -rf stories
mkdir stories
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1teTZXEE-9v5h2mn1SmuKyxlzw3RqNl9m' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1teTZXEE-9v5h2mn1SmuKyxlzw3RqNl9m" -O stories/train.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_Tt78pxV0kZFN2mAYtDwuWufe8CH4OFv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_Tt78pxV0kZFN2mAYtDwuWufe8CH4OFv" -O stories/valid.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nd5QFKd1LH70GWeBJrMsWyz0WGV6uec2' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nd5QFKd1LH70GWeBJrMsWyz0WGV6uec2" -O stories/test.txt.gz && rm -rf /tmp/cookies.txt

rm -rf abstracts
mkdir abstracts
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hXFjrgD_dwbMbkw-NkuGCaQXwdqF3XfS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hXFjrgD_dwbMbkw-NkuGCaQXwdqF3XfS" -O abstracts/train.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1au1diLKq5IeY0sXBqEqq0w0CMQ7CQ9de' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1au1diLKq5IeY0sXBqEqq0w0CMQ7CQ9de" -O abstracts/valid.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NX4EBz_xb6MZlrqmvX-uwy9LehCBBt7o' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NX4EBz_xb6MZlrqmvX-uwy9LehCBBt7o" -O abstracts/test.txt.gz && rm -rf /tmp/cookies.txt

rm -rf lyrics
mkdir lyrics
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MHTODeuTMmGLllVqeTfe3AhYg-r6UYlw' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MHTODeuTMmGLllVqeTfe3AhYg-r6UYlw" -O lyrics/train.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1f2TDWkzh8jaH9Jp1wWz1LQ0mBk4XjAuR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1f2TDWkzh8jaH9Jp1wWz1LQ0mBk4XjAuR" -O lyrics/valid.txt.gz && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1qoP_kceNxGwz-5IvKSKovEJxYy2u9MWC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1qoP_kceNxGwz-5IvKSKovEJxYy2u9MWC" -O lyrics/test.txt.gz && rm -rf /tmp/cookies.txt

gunzip */*.gz
