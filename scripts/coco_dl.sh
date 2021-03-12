# Credit to https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

dataset_root_path="./data"

# BEWARE: Symbolic links to a directory will also pass this check
# https://stackoverflow.com/questions/59838/how-can-i-check-if-a-directory-exists-in-a-bash-shell-script
if [ ! -d "$dataset_root_path" ]; then
    mkdir $dataset_root_path
fi

cd $dataset_root_path

#######################################
#     Download COCO2017 Images        #
#######################################
mkdir coco
cd coco

for imgs_zip in "train2017" "val2017" # "test2017" "unlabeled2017"
do
    echo "Downloading: http://images.cocodataset.org/zips/$imgs_zip.zip"
    curl -OL http://images.cocodataset.org/zips/$imgs_zip.zip
    echo "Unzipping: $imgs_zip.zip"
    unzip -q $imgs_zip.zip
    rm $imgs_zip.zip
done

#######################################
#    Download COCO2017 Annotations    #
#######################################
cd ../

for annos_zip in "annotations_trainval2017" "stuff_annotations_trainval2017" # "image_info_test2017" "image_info_unlabeled2017" 
do
    echo "Downloading: http://images.cocodataset.org/zips/$annos_zip.zip"
    curl -OL http://images.cocodataset.org/annotations/$annos_zip.zip
    echo "Unzipping: $annos_zip.zip"
    unzip -q $annos_zip.zip
    rm $annos_zip.zip
done
