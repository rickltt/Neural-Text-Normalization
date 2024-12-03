chmod a+x ./scripts/hfd.sh

export HF_ENDPOINT=https://hf-mirror.com

./scripts/hfd.sh google/electra-small-discriminator --tool aria2c -x 16 --include "*.bin" --local-dir ./models/electra-small-discriminator

./scripts/hfd.sh google-t5/t5-small --tool aria2c -x 16 --include "*.bin" --local-dir ./models/t5-small

