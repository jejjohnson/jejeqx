save_dir=$1
url_reference=https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/dc_data1/dc_ref.tar.gz


wget --directory-prefix=$save_dir $url_reference
tar -xvf $save_dir/dc_ref.tar.gz --directory=$save_dir
rm -f $save_dir/dc_ref.tar.gz