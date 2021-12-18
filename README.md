# Re-ResNet: Refactoring ResNets with Assembled Architecture Refinements and Improved Training Methods

## Requirements

```shell
pip install -U tensorflow==2.3.4
git clone https://github.com/sbl1996/hanser.git
cd hanser && pip install -e .
```

## Ablation study
All Re-ResNet variants for ablation study are in `nets.py`. To facilitate measurement of throughputs, we rewrite them in PyTorch. You may use the code to test the inference throughput on GPU.

## Training
We provide training scripts and logs for the 3 training recipes T1, T2, T3 on Re-ResNet. The pre-trained models can not be published directly now beacause of the policy of our institution. We hope that the provided training scripts and logs are enough for reference and reproduction. 