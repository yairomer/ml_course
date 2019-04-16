# Material for Course # [046195](https://ug3.technion.ac.il/rishum/course?MK=46195&CATINFO=&SEM=201802): Intro to Machine Learning (at The Technion)

This repository contains the workshops, homework assignment and some auxiliary material

[Web site](https://yairomer.github.io/ml_course/)

## Online Interactive Version Using MyBinder

The best way for viewing the note books in this repository is by using the following link:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/yairomer/ml_course/master) (Be patient, it takes the servers a few minutes to start)

## Repository content

``` text
├── workshops                   - The workshops' notebooks.
├── assignments                 - The assignments' notebooks.
├── html                        - A folder containing a static version (as HTML files) of the workshops
├── slides                      - The slides which are used in the workshops.
│   └── plugin
├── widgets                     - Some interactive widgets which are used in the course.
├── datasets                    - The cleaned up datasets used in the course.
│   ├── datasets_generation     - The notebooks used for generating the cleaned up datasets.
│   └── original                - A folder for placing the full original datasets used to create
│                                 the cleaned up datasets (is not push into the repository).
├── media                       - Images used in this repository.
├── generate.sh                 - A script for generating the static content.
├── css                         - Some CSS which is used in the repository.
│   └── style.css
├── lib                         - 3rd party resources
├── README.md
├── index.html
├── requirements.txt
└── postBuild
```

## Resources

- Christopher Bishop's book: Pattern Recognition and Machine Learning: [link](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)
- Andrew Ng's course at Stanford: [websites](http://cs229.stanford.edu/), [lecture videos](https://www.youtube.com/playlist?list=PLA89DCFA6ADACE599)
  - A review of probability theory from the course: [link](http://cs229.stanford.edu/section/cs229-prob.pdf)
- Statistical Learning Theory and Applications course at MIT: [website](http://www.mit.edu/~9.520/fall18/), [lectures videos](https://www.youtube.com/watch?list=PLyGKBDfnk-iCXhuP9W-BQ9q2RkEIA5I5f&v=Q5itLKscYTA)

## Serving web pages locally 

To serve the web pages in this repository locally run

```bash
python3 -m http.server 8080
```

## Contributing

- To commit Jupyter notebook without cell run numbers and metadata:
  1. Install jq

      ``` bash
      sudo apt install jq
      ```

  2. run the following command in the repository folder:

      ```bash
      git config --local include.path ../.gitconfig
      ```