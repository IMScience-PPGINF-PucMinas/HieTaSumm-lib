# hierarchical approach for video summarization

HieTaSumm — Hierarchical Time-aware Summarizer is a Python library for generating dynamic video summaries focusing on hierarchical graphs that are capable of processing multiple frames. The main features presented by HieTaSumm are the representation of videos in graphs that consider the importance and coherence of time to form clusters that represent keyframes or keyshots. By using hierarchical methods, the generated clusters guarantee that the cuts did not produce new locations and generate homogeneous regions. HieTaSumm's main target audience is students and professionals who want an accessible library to quickly experiment with, researchers developing new methods for processing videos.
HieTaSumm serves as a versatile toolkit for video summarization or video skimming, finding applications across diverse fields such as machine learning, data science, pattern analysis, and computer vision. Its generic nature makes it adaptable to a wide range of scenarios within these domains.

## Getting started

###Pre-build binaries
The Python package can be installed with Pypi:

pip install HieTaSumm

Supported systems:

Python 3.5, 3.6, 3.7, 3.8

Linux 64 bits, macOS, Windows 64 bits

With setuptools
The file setup.py is a thin wrapper around the cmake script. The following commands will download the library, create a binary wheel and install it with pip.

git clone link-para-github
cd HieTaSum
python setup.py

More information avalable on he full documentation: [HieTaSumm Documentation](https://hietasumm-doc.readthedocs.io/).

and about the funcionalites: The usage examples are shown on this link: [Usage Examples](https://hietasumm-doc.readthedocs.io/en/latest/examples.html)

## Citations
If you find this code useful for your research, consider cite our paper:
```
@INPROCEEDINGS{cardoso2021summarization,
    AUTHOR="Leonardo Vilela Cardoso and Zenilton Kleber Patrocínio Jr and Silvio Guimarães",
    TITLE="Hierarchical Time-aware Approach for Video Summarization",
    BOOKTITLE="BRACIS 2023",
    ADDRESS="Belo Horizonte, MG, Brazil",
    DAYS="25-29",
    MONTH="sep",
    YEAR="2023",
}
```

## Contact
Leonardo Vilela Cardoso with this e-mail: lvcardoso@sga.pucminas.br
