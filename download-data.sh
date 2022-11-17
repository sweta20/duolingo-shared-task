
# You can find out the available corpora for a pair of language using opus_get: https://github.com/Helsinki-NLP/OpusTools/tree/master/opustools_pkg

# For example
# Opus_get lists all the available resources for a language pair.
# opus_get -s en -t ja --list | grep OPUS | cut -f4 -d/ | cut -f2 -d- 
# opus_get -s en -t ja --list | grep OPUS | cut -f5 -d/  

# Parameters
src=en
tgt=ko
name=Tatoeba
version=v20190709
global_data_dir=data/$src-$tgt
mkdir -p $global_data_dir
cd $global_data_dir
wget -O $src-$tgt.txt.zip http://opus.nlpl.eu/download.php?f=${name}/${version}/moses/$src-$tgt.txt.zip 
unzip $src-$tgt.txt.zip 

rm $src-$tgt.txt.zip 
# cd ../../