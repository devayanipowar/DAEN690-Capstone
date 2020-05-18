#!/bin/bash

set -euo pipefail

help_message() {
	printf "${0}: Combines all the training and test folders into their respective locations\n\t-i /path/to/train \n\t-s /path/to/tier3 \n\t-x /path/to/test \n\t-l /path/to/holdout"
	
}

if [ $# -lt 3 ]; then 
        help_message
        exit 1
fi


while getopts "i:s:x:l:h" OPTION
do 
	case $OPTION in	
		h) 
			help_message
			exit 1
			;;
		i)
			train="$OPTARG"
			;;
		s)
			tier3="$OPTARG"
			;;
		x)
			tst="$OPTARG"
			;;
		l)
			hold="$OPTARG"
			;;
	esac
done

mkdir -p all_test
mkdir -p all_test/images
mkdir -p all_test/labels
mkdir -p all_test/targets
mkdir -p all_train
mkdir -p all_train/images
mkdir -p all_train/labels
mkdir -p all_train/targets

cp -R "$tst"/images/. all_test/images
cp -R "$tst"/labels/. all_test/labels
cp -R "$hold"/images/. all_test/images
cp -R "$hold"/labels/. all_test/labels

cp -R "$train"/images/.	all_train/images
cp -R "$train"/labels/. all_train/labels
cp -R "$tier3"/images/.	all_train/images
cp -R "$tier3"/labels/. all_train/labels


