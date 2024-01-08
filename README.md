# mbt_gym
`mbt_gym` is a module which provides a suite of gym environments for training reinforcement learning (RL) agents to solve model-based high-frequency trading problems such as market-making and optimal execution. The module is set up in an extensible way to allow the combination of different aspects of different models. It supports highly efficient implementations of vectorized environments to allow faster training of RL agents.

It includes gym environments for popular analytically tractable market making models, as well as more complex models that prove difficult to solve analytically.

The associated paper can be found at https://dl.acm.org/doi/pdf/10.1145/3604237.3626873 and https://arxiv.org/abs/2209.07823.

## Contributions are welcome!
If you wish to contribute to this repository, please read the details of how to do so in the 
[CONTRIBUTING.md](./CONTRIBUTING.md) file in the root directory of the repository. For ideas on code that you could 
contribute, please look at the [roadmap](./roadmap.md).  

## Using mbt_gym with Docker

To use the `mbt_gym` package from within a docker container (see [instructions on how to install docker](https://docs.docker.com/engine/install/ubuntu/))
, first change directory into the
docker subdirectory using `cd docker` and then follow the instructions below.

### Building

To build the container:

```
sh build_image.sh
```

### Running

Run the start container script (mounting ../, therefore mounting `mbt_gym`), and specify a port for jupyter notebook:

```
sh start_container.sh 8877
```

(Note: if you wish to add gpus to container, just add ```--gpus device=0``` to ```start_container.sh``` to use one gpu 
or ```--gpus all``` to add all gpus available.)

To work in the container via shell:

```
sh exec_mbt_gym.sh
```

## Citing mbt_gym

When using `mbt_gym`, please cite our [ACM ICAIF 2023 paper](https://arxiv.org/abs/2209.07823) by using the following
BibTeX entry:
```
@inproceedings{JeromeSSH23,
  author       = {Joseph Jerome and
                  Leandro S{\'{a}}nchez{-}Betancourt and
                  Rahul Savani and
                  Martin Herdegen},
  title        = {Mbt-gym: Reinforcement learning for model-based limit order book trading},
  booktitle    = {4th {ACM} International Conference on {AI} in Finance, {ICAIF} 2023,
                  Brooklyn, NY, USA, November 27-29, 2023},
  pages        = {619--627},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3604237.3626873},
  doi          = {10.1145/3604237.3626873},
  note         = {arXiv preprint arXiv:2209.07823}
}
```
