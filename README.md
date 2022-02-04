## Auto-encoding Bilinear Classifier (ABC)

This is the code for "Twitter Spam Detection via Bilinear Autoencoding Reconstruction Error"

Two datasets are used for evaluation. Here are the official websites and references, please use under the official license :

- Social Honeypot Dataset: https://infolab.tamu.edu/data/#social-honeypot-dataset

> K. Lee, J. Caverlee, and S. Webb, “Uncovering social spammers: social honeypots+ machine learning,” in *Proceedings of the 33rd International ACM SIGIR Conference on Research and Development in Information Retrieval*, 2010, pp. 435–442.

- 6 Million Spam Tweets Dataset: http://nsclab.org/nsclab/resources/ 

> C. Chen, J. Zhang, X. Chen, Y. Xiang, and W. Zhou, “6 million spam tweets: A large ground truth for timely Twitter spam detection,” in *Proceedings of the 2015 IEEE International Conference on Communications*, 2015, pp. 7065–7070.

We provide the code for preprocessing and extending the features of honeypot dataset. A simple running script is in `run_main.py`, including some baselines from SciKit-Learn module.

