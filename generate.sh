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

## Slide to pdf
# `npm bin`/decktape reveal -s 1440x900 http://localhost:8080/slides/workshop.html?id=01 /tmp/workshop_01.pdf 