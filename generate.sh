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

## Workshop 6
jupyter nbconvert --to html --output-dir=$repo_dir/html/ $repo_dir/workshops/workshop_07.ipynb

## Slide to pdf
# `npm bin`/decktape reveal -s 1440x900 http://localhost:8080/slides/workshop.html?id=01 /tmp/workshop_01.pdf 

## Assignment 1
jq --indent 1 '(.cells[] | select(has("outputs")) | .outputs) = [] | .cells[].metadata = {} | ((.cells[] | select(has("execution_count")) | .execution_count) = null)' $repo_dir/assignments/homework_1.ipynb > $repo_dir/assignments/homework_1.ipynb.tmp
mv $repo_dir/assignments/homework_1.ipynb.tmp $repo_dir/assignments/homework_1.ipynb