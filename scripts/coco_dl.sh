# Adapted from https://gist.github.com/mkocabas/a6177fc00315403d31572e17700d7fd9

dataset_root_path="./data"

if [[ $# >= 1 ]] ; then
    dataset_root_path=$1
fi

enable_unzip=true
download_all=false
remove_zip=true

# BEWARE: Symbolic links to a directory will also pass this check
# https://stackoverflow.com/questions/59838/how-can-i-check-if-a-directory-exists-in-a-bash-shell-script
if [ ! -d "$dataset_root_path" ]; then
    mkdir $dataset_root_path
fi

cd $dataset_root_path

#######################################
#             Zip Names               #
#######################################
arrCocoImg=("train2017" "val2017")
arrCocoAnno=("annotations_trainval2017" "stuff_annotations_trainval2017")

if [ "$download_all" = true ] ; then
    echo 'Download entire COCO dataset.'
    arrCocoImg+=("test2017")
    arrCocoImg+=("unlabeled2017")

    arrCocoAnno+=("image_info_test2017")
    arrCocoAnno+=("image_info_unlabeled2017")
fi

#######################################
#     Download COCO2017 Images        #
#######################################

# if the second argument is "no_image" then we want the images.
if [[ $2 != "no_image" ]]; then 
    if [ ! -d "coco" ]; then
        mkdir coco
    fi

    cd coco

    for imgs_zip in "${arrCocoImg[@]}"
    do
        if [ ! -f $imgs_zip.zip -a ! -d $imgs_zip ]; then
            echo "$imgs_zip.zip and $imgs_zip/ not found!"
            echo "Downloading from... http://images.cocodataset.org/zips/$imgs_zip.zip"
            curl -OL http://images.cocodataset.org/zips/$imgs_zip.zip
        fi
        
        if [ -f $imgs_zip.zip ]; then
            if [ "$enable_unzip" = true ] ; then
                echo "Unzipping: $imgs_zip.zip"
                unzip -q $imgs_zip.zip
            fi

            if [ "$remove_zip" = true ] ; then
                echo "Deleting file: $imgs_zip.zip"
                rm $imgs_zip.zip
            fi
        else
            echo "$imgs_zip.zip not found. Ignoring zip-related instructions..."
        fi
    done

# Moved this as its only needed if Image dataset is downloaded
cd ../

fi

#######################################
#    Download COCO2017 Annotations    #
#######################################

for annos_zip in "${arrCocoAnno[@]}"
do
    if [ ! -f $annos_zip.zip ]; then
        echo "$annos_zip.zip not found!"
        echo "Downloading from... http://images.cocodataset.org/zips/$annos_zip.zip"
        curl -OL http://images.cocodataset.org/annotations/$annos_zip.zip
    fi

    if [ -f $annos_zip.zip ]; then
        if [ "$enable_unzip" = true ] ; then
            echo "Unzipping: $annos_zip.zip"
            unzip -q $annos_zip.zip
        fi

        if [ "$remove_zip" = true ] ; then
            echo "Deleting file: $annos_zip.zip"
            rm $annos_zip.zip
        fi
    else
        echo "$annos_zip.zip not found. Ignoring zip-related instructions..."
    fi
done
