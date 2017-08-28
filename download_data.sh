glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'
infersentpath='https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle'

# GloVe
echo "Getting GloVe from " $glovepath
mkdir pretrained/GloVe
curl -LO $glovepath
$ZIPTOOL glove.840B.300d.zip -d pretrained/GloVe
rm glove.840B.300d.zip

# infersent
echo "Getting InferSent model from " $infersentpath
curl -Lo encoders/infersent/infersent.allnli.pickle $infersentpath
