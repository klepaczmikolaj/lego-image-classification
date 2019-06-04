#!/usr/bin/env bash
# Specify input directory name
if [[ "$#" -ne 1 ]]; then
    echo "Specify input directory, eg: input_data (without slash)"
    exit 1
fi

input_dir=$1
output_dir=${input_dir}_trim

# clear output_dir if exists
if [[ -d "$output_dir" ]]; then rm -Rf $output_dir; fi

cp -r ${input_dir} ${output_dir}
sub_dirs=($(ls ${output_dir}))

min_cnt=10000000
# get min count of images
for el in ${sub_dirs[*]}; do
    count=$(ls $output_dir/$el | wc -l)
    if (( $count < $min_cnt )); then
        min_cnt=$count
    fi
done

echo "Min count: ${min_cnt}"

min_cnt=$((min_cnt-1))

# trim files
for el in ${sub_dirs[*]}; do
    ls $output_dir/$el/* | head -n -${min_cnt} | xargs rm
done
