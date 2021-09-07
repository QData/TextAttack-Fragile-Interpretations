# TextAttack-Fragile-Interpretations
Code for the paper: ["Perturbing Inputs for Fragile Interpretations in Deep Natural Language Processing"](https://arxiv.org/abs/2108.04990)
(EMNLP BlackboxNLP - 2021)

Pre-calculated candidates and interpretations are available on Google drive [here](https://drive.google.com/drive/folders/1U_bcpKa9OHR11z_o1EXo1QPzUcOxs5jT?usp=sharing). The results can be replicated by running the `results-metric.py` script. The exact commmands are detailed in Step-5.

We strongly recommend using `conda` to manage dependencies.

Run `conda create -n frag-exp python=3.6` and subsequently `conda activate frag-exp`.

Run `pip install -r requirements.txt`


Following steps re-run the candidate generation process and re-calculate interpretations.

1. Install `Textattack` from the `TextAttack` folder's `dist` folder  by installing the wheel: 
`pip install Textattack/dist/textattack-0.2.14-py3-none-any.whl`

2. Run `python generate_candidates.py --model=distilbert --dataset=sst2 --number=500 --split=validation`. All options can be edited for different datasets and models. By default save paths are `./candidates`. 

3.  Run `python calculate_interpretations.py --model=distilbert --dataset=sst2 --interpretmethod=IG --number=500 --split=validation`. All options can be edited for different datasets and models. By default save paths are `./interpretations`. 

4. Once all interpretations have been calculated, run `python results-metrics.py --model=distilbert --dataset=sst2 --interpretmethod=IG --number=500 --split=validation --metric=rkc`.

The available metrics are `rkc (Rank Correlation)`, `topk (Top-K Intersection)`,`ppl (Perplexity)`, `grm (Grammar errors)` and `conf (Model Confidence)`. Results are stored in `./results`.
