#!/usr/bin/env bash
# Specify input directory name
if [[ "$#" -ne 1 ]]; then
    echo "Specify input directory, eg: input_data (without slash)"
    exit 1
fi

input_dir=$1
output_dir=${input_dir}_trim
test_dir=${input_dir}_test

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

min_cnt=$((min_cnt-1))

echo "Min count: ${min_cnt}"

# trim files
for el in ${sub_dirs[*]}; do
    ls $output_dir/$el/* | head -n -${min_cnt} | xargs rm
done

# clear test_dir if exists
if [[ -d "$test_dir" ]]; then rm -Rf $test_dir; fi
mkdir $test_dir

# move 10% of picture files to test dir
test_cnt=$((min_cnt/10))
for sub_dir in ${sub_dirs[*]}; do
    mkdir ${test_dir}/${sub_dir}
    shuf -zn${test_cnt} -e ${output_dir}/$sub_dir/*.jpg | xargs -0 mv -t $test_dir/${sub_dir}
done
