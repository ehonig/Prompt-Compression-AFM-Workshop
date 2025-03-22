# Attention-Only Compressor

This repository contains code for the NeurIPS 2024 Adaptive Foundation Models Workshop paper [Better Prompt Compression Without Multi-Layer
Perceptrons](https://arxiv.org/abs/2501.06730).

## Setup

<!-- The trainer.py patch can be dealt with in other ways, but it is left as a memento :) -->

```bash
# Install uv if necessary
which uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup Python environment
uv venv --python=3.10
source .venv/bin/activate
uv pip install torch==2.4.0 peft==0.11.1 transformers==4.43.1 nltk==3.9.1 rouge==1.0.1 wandb==0.17.8 pandas==2.1.4

# patch `transformers/trainer.py` to use `save_model` when `save_file` fails
sed -i '
/if self\.args\.save_safetensors:/,/torch\.save(state_dict, os\.path\.join(output_dir, WEIGHTS_NAME))/c\
                if self.args.save_safetensors:\
                    try:\
                        safetensors.torch.save_file(\
                            state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}\
                        )\
                    except: # patch this to fix model saving, no need to fork\
                        safetensors.torch.save_model(\
                            self.model, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}\
                        )\
                else:\
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
' "$(uv pip show transformers | grep Location | awk -F': ' '{print $2}')/transformers/trainer.py"

# download and setup arxiv abstracts datasets, note that this performs minimal preprocessing
cd data
curl -Lo arxiv.zip https://www.kaggle.com/api/v1/datasets/download/Cornell-University/arxiv
unzip arxiv.zip
python get_arxiv_data.py
cd ..
```

## Usage

Run `pretrain_and_eval.py` from this directory, specifying arguments as necessary. For example:

```bash
python pretrain_and_eval.py --debug=True --LLM=AOC --max_length=96 --num_mem=16 --learning_rate=2e-4
```

## Questions

Please open an issue or email e.honig@ucla.edu with any questions or comments regarding this code.

Thank you for your interest in our work, which is heavily influenced by two key predecessors:

1. [500xCompressor](https://arxiv.org/abs/2408.03094): Our work and codebase is based on their [excellent code](https://github.com/ZongqianLi/500xCompressor).
2. [In-context Autoencoder](https://arxiv.org/abs/2307.06945): Our work follows the direction of training an autoencoder using a frozen LLM decoder.

There exist many more related works on prompt compression and training language autoencoders.  
We include the two top influences.

## Cite

```
@inproceedings{honig2024better,
      title={Better Prompt Compression Without Multi-Layer Perceptrons}, 
      author={Edouardo Honig and Andrew Lizarraga and Zijun Frank Zhang and Ying Nian Wu},
      booktitle={Advances in Neural Information Processing Systems: Adaptive Foundation Models Workshop},
      year={2024},
      url={https://arxiv.org/abs/2501.06730}, 
}
@misc{li2024500xcompressor,
      title={500xCompressor: Generalized Prompt Compression for Large Language Models}, 
      author={Zongqian Li and Yixuan Su and Nigel Collier},
      year={2024},
      eprint={2408.03094},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.03094}, 
}
@inproceedings{
  ge2024incontext,
  title={In-context Autoencoder for Context Compression in a Large Language Model},
  author={Tao Ge and Hu Jing and Lei Wang and Xun Wang and Si-Qing Chen and Furu Wei},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=uREj4ZuGJE}
}
```
