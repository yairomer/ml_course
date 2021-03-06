#!/bin/bash
repo_dir="$( cd $( dirname "${BASH_SOURCE[0]}" ) && pwd )"

## Workshop 1
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_01.ipynb

## Workshop 2
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_02.ipynb

## Workshop 3
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_03.ipynb

## Workshop 4
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_04.ipynb

## Workshop 5
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_05.ipynb

## Workshop 6
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_06.ipynb

## Workshop 7
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_07.ipynb

## Workshop 8
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_08.ipynb

## Workshop 9
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_09.ipynb

## Workshop 10
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_10.ipynb

## Workshop 11
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_11.ipynb

## Workshop 12
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_12.ipynb

## Slide to pdf
# `npm bin`/decktape reveal -s 1440x900 http://localhost:8080/slides/workshop.html?id=01 /tmp/workshop_01.pdf 

## Assignment 1
jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = [] | .cells[].metadata = {} | ((.cells[] | select(has("execution_count")) | .execution_count) = null)' $repo_dir/assignments/homework_1.ipynb > $repo_dir/assignments/homework_1.ipynb.tmp
mv $repo_dir/assignments/homework_1.ipynb.tmp $repo_dir/assignments/homework_1.ipynb
jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = [] | .cells[].metadata = {} | ((.cells[] | select(has("execution_count")) | .execution_count) = null)' $repo_dir/assignments/homework_2.ipynb > $repo_dir/assignments/homework_2.ipynb.tmp
mv $repo_dir/assignments/homework_2.ipynb.tmp $repo_dir/assignments/homework_2.ipynb
jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = [] | .cells[].metadata = {} | ((.cells[] | select(has("execution_count")) | .execution_count) = null)' $repo_dir/assignments/homework_3.ipynb > $repo_dir/assignments/homework_3.ipynb.tmp
mv $repo_dir/assignments/homework_3.ipynb.tmp $repo_dir/assignments/homework_3.ipynb
